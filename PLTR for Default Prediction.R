# ==============================================================================
# Penalised Logistic Tree Regression (PLTR) for Credit Scoring
# Objective: Extract non-linear default cliffs using scout trees and Adaptive Lasso
# ==============================================================================

rm(list = ls())

# Load Required Libraries
library(dplyr)
library(rpart)
library(rpart.plot)
library(glmnet)
library(pROC)
library(ggplot2)
library(randomForest)

# ==============================================================================
# 1. Data Loading & Feature Engineering (Altman Z-Score Ratios)
# ==============================================================================
df <- read.csv("combined_data.csv")

clean_names <- c("Ticker", "Name", "Country", "MarketCap", "TotalAssets", 
                 "RetainedEarnings", "EBIT", "TotalLiabilities", "Revenue", 
                 "WorkingCapital", "StatusCode", "Sector", "DelistedFlag", 
                 "Rating", "StarMinePD", "Region")
colnames(df) <- clean_names

# Create target variable (Default)
df$Default <- ifelse(df$DelistedFlag == TRUE, 1, 0)

# Engineer Structural Economic Pillars
df <- df %>%
  mutate(
    Liquidity = WorkingCapital / TotalAssets,
    RetainedEarningsRatio = RetainedEarnings / TotalAssets,
    EBITRatio = EBIT / TotalAssets,
    MarketLeverage = MarketCap / TotalLiabilities,
    AssetTurnover = Revenue / TotalAssets
  )

# Subset and clean NA/Infinite values
base_vars <- c("Liquidity", "RetainedEarningsRatio", "EBITRatio", "MarketLeverage", "AssetTurnover")
df_clean <- df %>% select(Default, all_of(base_vars)) %>% na.omit()
df_clean <- df_clean[is.finite(rowSums(df_clean[, base_vars])), ]

# ==============================================================================
# 2. Scout Trees: Extracting Non-Linear Thresholds
# ==============================================================================
rule_matrix_list <- list()

# Extract Univariate Rules (1-split)
for (var in base_vars) {
  form <- reformulate(termlabels = var, response = "Default")
  uni_var_model <- rpart(formula = form, data = df_clean, 
                         control = rpart.control(maxdepth = 1, cp = -1, minsplit = 2))
  
  leaf_nodes <- as.factor(uni_var_model$where)
  if (length(levels(leaf_nodes)) > 1) {
    dummies <- model.matrix(~ leaf_nodes - 1)
    colnames(dummies) <- paste0(var, "_Node", colnames(dummies))
    rule_matrix_list[[var]] <- dummies
  }
}

# Extract Bivariate Rules (2-split)
for (i in 1:(length(base_vars) - 1)) {
  for (j in (i + 1):length(base_vars)) {
    var1 <- base_vars[i]
    var2 <- base_vars[j]
    pair_name <- paste(var1, var2, sep = "_")
    
    form <- reformulate(termlabels = c(var1, var2), response = "Default")
    bi_var_model <- rpart(formula = form, data = df_clean, 
                          control = rpart.control(maxdepth = 2, cp = -1, minsplit = 2))
    
    leaf_nodes <- as.factor(bi_var_model$where)
    if (length(levels(leaf_nodes)) > 1) {
      dummies <- model.matrix(~ leaf_nodes - 1)
      colnames(dummies) <- paste0(pair_name, "_Node", colnames(dummies))
      rule_matrix_list[[pair_name]] <- dummies
    }
  }
}

# Combine base variables with all 56 extracted rule features
df_final <- cbind(df_clean, do.call(cbind, rule_matrix_list))

# ==============================================================================
# 3. The Filter: Adaptive Lasso
# ==============================================================================
y <- df_final$Default
x <- as.matrix(df_final[, names(df_final) != "Default"])

# Ridge Regression to calculate adaptive penalty weights
set.seed(42)
ridge_model <- cv.glmnet(x, y, family = "binomial", alpha = 0)
ridge_coefs <- coef(ridge_model, s = "lambda.min")[-1, ]
penalty_weights <- 1 / abs(ridge_coefs)

# Adaptive Lasso Execution
set.seed(42)
adaptive_lasso <- cv.glmnet(x, y, family = "binomial", alpha = 1, penalty.factor = penalty_weights)

# ==============================================================================
# 4. Model Showdown: Logit vs. Random Forest vs. PLTR
# ==============================================================================
actuals <- df_clean$Default
x_pltr <- as.matrix(cbind(df_clean[, base_vars], df_final[rownames(df_clean), grepl("_Node", colnames(df_final))]))

# Train Baseline Logit
baseline_model <- glm(Default ~ Liquidity + RetainedEarningsRatio + EBITRatio + MarketLeverage + AssetTurnover, 
                      data = df_clean, family = "binomial")

# Train Random Forest
df_clean$Default_Factor <- as.factor(df_clean$Default)
set.seed(42)
rf_model <- randomForest(Default_Factor ~ Liquidity + RetainedEarningsRatio + EBITRatio + MarketLeverage + AssetTurnover, 
                         data = df_clean, ntree = 500)

# Generate Predictions
prob_baseline <- predict(baseline_model, type = "response")
prob_rf <- predict(rf_model, type = "prob")[, "1"]
prob_pltr <- predict(adaptive_lasso, newx = x_pltr, s = "lambda.min", type = "response")

# ROC & AUC Calculation
roc_baseline <- roc(actuals, prob_baseline, quiet = TRUE)
roc_rf <- roc(actuals, prob_rf, quiet = TRUE)
roc_pltr <- roc(actuals, as.numeric(prob_pltr), quiet = TRUE)

# Brier Scores
brier_baseline <- mean((prob_baseline - actuals)^2)
brier_rf <- mean((prob_rf - actuals)^2)
brier_pltr <- mean((as.numeric(prob_pltr) - actuals)^2)

# Master Showdown Plot
plot(roc_baseline, col = "blue", main = "The Accuracy vs Interpretability Tradeoff", lwd = 2)
lines(roc_rf, col = "green", lwd = 2)
lines(roc_pltr, col = "red", lwd = 2)
legend("bottomright", 
       legend = c(paste("Baseline Logit (AUC:", round(auc(roc_baseline), 3), ")"),
                  paste("Random Forest (AUC:", round(auc(roc_rf), 3), ")"),
                  paste("PLTR Hybrid (AUC:", round(auc(roc_pltr), 3), ")")), 
       col = c("blue", "green", "red"), lwd = 2)

# ==============================================================================
# 5. Auditing the Black Box: Extracting the Final Scorecard
# ==============================================================================
coef_matrix <- as.matrix(coef(adaptive_lasso, s = "lambda.min"))
surviving_vars <- data.frame(Feature = rownames(coef_matrix), Coefficient = coef_matrix[, 1])

# Isolate top non-linear rules surviving the Lasso penalty
top_rules <- surviving_vars %>%
  filter(Coefficient != 0 & grepl("_Node", Feature)) %>%
  arrange(desc(abs(Coefficient)))

# Visualizing the Top Penalties
chart_data <- head(top_rules, 10)
chart_data$Clean_Name <- gsub("_Nodeleaf_nodes", " Node ", chart_data$Feature)

ggplot(chart_data, aes(x = reorder(Clean_Name, Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity", fill = "steelblue", color = "black") +
  coord_flip() +
  labs(title = "The PLTR Scorecard: Top Adaptive Lasso Penalties",
       x = "Extracted Non-Linear Rule",
       y = "Penalty Coefficient (Impact on Default Risk)") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 9, face = "bold"),
        plot.title = element_text(size = 14, face = "bold"))

# ==============================================================================
# 6. Visualizing the Winning Scout Trees
# ==============================================================================
tree_formulas <- list(
  "AssetTurnover + EBITRatio",
  "AssetTurnover + RetainedEarningsRatio",
  "AssetTurnover + MarketLeverage",
  "MarketLeverage + Liquidity"
)

par(mfrow = c(2, 2))
for (form in tree_formulas) {
  tree_model <- rpart(as.formula(paste("as.factor(Default) ~", form)), 
                      data = df_clean, method = "class",
                      control = rpart.control(maxdepth = 2, cp = -1, minsplit = 2))
  
  rpart.plot(tree_model, type = 4, extra = 106, 
             main = paste("Scout Tree:", form),
             box.palette = c("palegreen3", "tomato"),
             branch.lty = 3, shadow.col = "gray", nn = TRUE)
}
par(mfrow = c(1, 1))