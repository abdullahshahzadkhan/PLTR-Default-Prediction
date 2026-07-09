# Penalised Logistic Tree Regression: Algorithmic Credit Scoring

➡️ **[Read the Full Interactive Report & Code Execution][https://abdullahshahzadkhan.github.io/PLTR-Default-Prediction/]**

## 📌 Project Overview
This project implements a **Penalised Logistic Tree Regression (PLTR)** engine to optimize corporate credit risk scoring. Using financial data from European non-financial corporations (sourced via Refinitiv), the objective was to bridge the gap between highly accurate but uninterpretable "black-box" models (like Random Forests) and highly interpretable but rigid linear models (like Logistic Regression).

By utilizing short-depth decision trees to detect extreme risk thresholds and applying an Adaptive Lasso to filter noise, the model successfully condensed complex, non-linear risk profiles into a fully auditable scorecard of just 19 rules.

## 💼 The Business Case & Challenges
* **The "Default Cliff":** Corporate defaults rarely happen on a smooth, linear scale. Companies often survive declining health until multiple interacting metrics break simultaneously (e.g., negative operating profit combined with severe illiquidity). Standard linear models average out these breaks, missing the "cliffs."
* **The Dilution Trap:** In this dataset, the default rate is a severely imbalanced **2.8%**. Black-box models like Random Forests average their predictions across hundreds of trees. This averaging effect dilutes rare, sharp warning signals into weak probabilities.
* **Regulatory Compliance:** Financial institutions require "white-box" models. A model cannot simply predict a default; it must legally explain *why* the borrower was flagged.

## 🛠️ Technical Methodology
1. **Feature Engineering:** Purged financial/shell corporations and transformed raw balance sheet metrics into the 5 structural **Altman Z-Score pillars** to prevent spurious machine learning correlations.
2. **Automated Threshold Detection (Scout Trees):** Deployed 1-split and 2-split decision trees to hunt for absolute risk cliffs, extracting **56 unique binary rules** (e.g., 100% default probability clusters).
3. **Adaptive Lasso Filtering:** Fed all 56 overlapping rules into an Adaptive Lasso algorithm. The mathematical penalty ruthlessly shrank weak/noisy rules to a coefficient of exactly $0.00$, leaving only **19 powerful, independent predictors**.
4. **Scorecard Generation:** Converted the surviving non-linear rules into a simplified, interpretable risk-weighting scorecard for compliance auditing.

## 📊 Key Results
* **Accuracy vs. Interpretability:** The PLTR Hybrid model successfully outperformed both the standard Baseline Logit and the Random Forest benchmark in discriminatory power.
  * **Baseline Logit AUC:** 0.603
  * **Random Forest AUC:** 0.655
  * **PLTR (Hybrid) AUC:** **0.713**
* **The "Double-Cut" Discovery:** Even though the algorithm was permitted to build interaction trees (mixing metrics like profitability and sales turnover), empirical results proved that a pure, double-threshold cut on operating profit alone was a deadlier signal than any interaction. The model learned precisely what drives failure at the absolute bottom of corporate solvency.

## 💻 Tech Stack
* **Language:** R
* **Core Libraries:** `glmnet` (Adaptive Lasso), `rpart` (Decision Trees), `pROC` (Model Evaluation), `dplyr`, `ggplot2`
