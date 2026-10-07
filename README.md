# Adult Census Income Prediction using C5.0 (Python & R Integration)

This repository contains an end-to-end machine learning pipeline that predicts whether an individual's annual income exceeds $50K/yr based on demographic and financial census data. 

The project implements the **C5.0 Decision Tree algorithm**. Because the most robust implementation of C5.0 exists in R, this Python script uses `rpy2` to seamlessly execute R's `C50` and `partykit` packages within a Python environment.

---

## Project Workflow (CRISP-DM Methodology)

The pipeline is structured following the Cross-Industry Standard Process for Data Mining (CRISP-DM) framework to ensure a clear, reproducible, and robust workflow.

### 1. Business Understanding
* **Objective:** Predict if a person's annual income falls into the high-income bracket (`>50K`) or base-income bracket (`<=50K`) using census attributes.
* **Use Case:** Socio-economic analysis, targeted financial services, and identifying key demographic drivers of wealth distribution.
* **Success Criteria:** Achieving robust predictive accuracy (>85%) while maintaining high model interpretability through a transparent decision tree structure.

### 2. Data Understanding
* **Data Source:** [UCI Adult Income Dataset](https://archive.ics.uci.edu/dataset/2/adult).
* **Collection:** Automated online fetching via Python's `requests` library directly from UCI databases (with GitHub mirrors as fallback).
* **Attributes:** 14 features spanning demographics (age, sex, race, native_country), education (education, education_num), occupation (workclass, occupation, relationship), and financial status (capital_gain, capital_loss, hours_per_week), with `income` as the target label.

### 3. Data Preparation
* **Data Cleaning:** Handled via `pandas`. Rows containing missing values (`?` or `nan`) were dropped, reducing the dataset from 48,842 to a clean total of **45,222 rows**.
* **Normalization:** Whitespaces in string features were stripped, and trailing periods (`.`) in test labels were removed for uniformity.
* **Data Splitting:** Using `scikit-learn`, the dataset was divided into:
  * **Training Data:** 75% (**33,916 cases**) used for model construction.
  * **Testing Data:** 25% (**11,306 cases**) reserved for unbiased out-of-sample evaluation.
  * **Stratification:** Applied `stratify=y` to preserve the natural class imbalance ratio across both splits.

### 4. Modeling
* **Algorithm:** C5.0 Decision Tree Classifier (executed via R's `C50` package).
* **Hyperparameters:** Configured with `trials = 1` (single tree configuration optimized for clear graphical interpretation), `minCases = 10`, and `CF = 0.1` for pruning.
* **Visualization:** Automatically renders a detailed graphical tree into a high-resolution PDF file (`c50_tree_visualization.pdf`) using the R `partykit` package.

---

## Program Output & Evaluation Results

When executed, the program successfully trains the model on the training set and evaluates it against the unseen testing dataset. Below is the detailed breakdown of the output and metrics:

### 1. Training Performance & Attribute Usage
* **Tree Complexity:** The resulting decision tree has a size of **34 terminal nodes (leaves)** with a training error rate of **13.1%** (4,435 misclassifications out of 33,916 training cases).
* **Feature Importance (Attribute Usage):** The C5.0 algorithm evaluated feature relevance across splits as follows:
  1. `capital_gain`: **100.00%** (Primary splitting driver)
  2. `relationship`: **95.62%**
  3. `capital_loss`: **95.25%**
  4. `education_num`: **39.71%**
  5. `age`: **32.53%**
  6. `occupation`: **30.23%**
  7. `hours_per_week`: **27.81%**
  8. `workclass`: **18.78%**

### 2. Test Set Evaluation Metrics (11,306 Cases)
* **Overall Accuracy:** **86.09%** (indicating strong generalizability on unseen data).
* **Classification Report:**
  * **Class `<=50K`:** Precision = `0.8801`, Recall = `0.9436`, F1-Score = `0.9107` (Support: 8,504)
  * **Class `>50K`:** Precision = `0.7807`, Recall = `0.6099`, F1-Score = `0.6848` (Support: 2,802)
* **Confusion Matrix:**
  * True Negatives (`<=50K` correctly predicted): **8,024**
  * False Positives (`<=50K` misclassified as `>50K`): **480**
  * False Negatives (`>50K` misclassified as `<=50K`): **1,093**
  * True Positives (`>50K` correctly predicted): **1,709**
* **ROC-AUC Score:** **0.8805**, confirming excellent discriminative capability between the two income classes.

---

## Decision Tree Visualization Analysis (`c50_tree_visualization.pdf`)

The generated PDF file visually breaks down the logic paths of the 34 terminal nodes:
* **Root Node Split:** The root splits directly on **`capital_gain > 6849`**. Individuals exceeding this capital gain threshold are classified immediately into the `>50K` bracket.
* **Secondary Branching:** For lower capital gains, the tree splits based on **`relationship`** status (e.g., distinguishing between family structures like husbands/wives versus single/unmarried individuals).
* **Deep Subtrees:** The left and central branches descend deeply into secondary features like **`education_num`**, **`age`**, **`occupation`**, and **`hours_per_week`**. For instance, individuals with higher education years (`education_num > 12`) combined with sustained weekly working hours (`hours_per_week > 30`) and specific managerial/professional occupations show higher probabilities of crossing the $50K threshold.
* **Terminal Leaf Probabilities:** Each leaf node in the PDF contains a probability bar chart representing the model's confidence distribution (`<=50K` vs `>50K`) for subsets falling into that specific rule.

---

## Machine Learning Conclusions

1. **Financial Indicators Dominate Wealth Classification:** `capital_gain` and `capital_loss` serve as the absolute strongest predictors for earning above $50K/year, appearing in over 95% of model splitting decisions.
2. **Socio-Demographic Interaction:** Family structure (`relationship`) and educational attainment (`education_num`) act as critical structural filters that shape baseline earning potential before occupation and weekly workload are factored in.
3. **Model Reliability:** With an accuracy of **86.09%** and an AUC of **0.8805**, the C5.0 decision tree proves to be a highly effective, transparent classifier for demographic data without requiring black-box complexity.

---

## How to Run

### Prerequisites
* **Python Libraries:** `pandas`, `numpy`, `requests`, `scikit-learn`, `rpy2`
* **R Packages:** `C50`, `partykit`

### References
* **Kohavi, R. (1996). Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid.**
* **[UCI Machine Learning Repository - Adult Dataset.](https://archive.ics.uci.edu/dataset/2/adult)**
* **[C5.0 R Package Documentation.](https://cran.r-project.org/web/packages/C50/index.html)**
