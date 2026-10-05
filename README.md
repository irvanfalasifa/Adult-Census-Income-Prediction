# Adult Census Income Prediction using C5.0 (Python & R Integration)

This repository contains an end-to-end machine learning pipeline that predicts whether an individual's annual income exceeds $50K/yr based on demographic and financial census data. 

The project implements the **C5.0 Decision Tree algorithm**. Because the most robust implementation of C5.0 exists in R, this Python script uses `rpy2` to seamlessly execute R's `C50` and `partykit` packages within a Python environment.

## Project Workflow (CRISP-DM Methodology)

The pipeline is structured following the Cross-Industry Standard Process for Data Mining (CRISP-DM) framework to ensure a clear, reproducible, and robust workflow.

### 1. Business Understanding
* **Objective:** Predict if a person's income falls into the `>50K` or `<=50K` bracket.
* **Use Case:** Demographic analysis, targeted marketing, and understanding socio-economic factors driving income levels.
* **Success Criteria:** Achieving high predictive accuracy and generating a highly interpretable decision tree that explains the splitting rules clearly.

### 2. Data Understanding
* **Data Source:** [UCI Adult Income Dataset](https://archive.ics.uci.edu/dataset/2/adult).
* **Collection Method:** The script automates data retrieval by fetching the raw data directly from UCI's online databases (with GitHub mirrors as a fallback) using the `requests` library. No manual downloading is required.
* **Features:** 14 attributes including demographic (age, sex, race, native_country), educational (education, education_num), occupational (workclass, occupation), and financial (capital_gain, capital_loss, hours_per_week) data.

### 3. Data Preparation
Data cleaning and preprocessing are handled natively in Python using `pandas` before passing the data to R:
* **Standardization:** Trims leading/trailing whitespaces from all string (object) columns and removes the trailing period (`.`) from the test set labels to ensure consistency between train and test splits.
* **Missing Value Handling:** Identifies unknown values (often represented as `?` or `nan` in this dataset), converts them to true `NaN` objects, and drops the incomplete rows to maintain data integrity.
* **Data Type Casting:** Enforces strict numeric types for continuous variables (`age`, `fnlwgt`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`) and converts categorical features to strings compatible with R factors.
* **Data Splitting:** Splits the cleaned dataset into **75% Training Data** and **25% Testing Data** using `scikit-learn`'s `train_test_split`. The split uses `stratify=y` to ensure the income class distribution remains balanced across both sets.

### 4. Modeling
* **Algorithm:** C5.0 Classification Model.
* **Integration:** Python passes the preprocessed DataFrames to the R environment using `rpy2` local converters. 
* **Hyperparameters:** Configurable via command-line arguments (CLI), including:
  * `trials`: Number of boosting iterations (default is 1 for a clean, single-tree visualization).
  * `minCases`: Minimum number of samples required to split a node.
  * `CF`: Confidence factor for pruning.
  * `winnow`: Attribute selection toggle.
* **Visualization:** The model automatically renders the generated decision tree into a high-resolution PDF file (`c50_tree_visualization.pdf`) directly from the R environment using the `partykit` plotting engine.

### 5. Evaluation
The R model predicts classes and probabilities for the test set, which are then passed back to Python for evaluation using `scikit-learn`. The script outputs:
* **Accuracy Score:** Overall correctness of the model.
* **Classification Report:** Detailed Precision, Recall, and F1-Scores for both `<=50K` and `>50K` classes.
* **Confusion Matrix:** True Positives, True Negatives, False Positives, and False Negatives.
* **ROC-AUC Score:** Evaluates the model's ability to distinguish between the two classes based on predicted probabilities.

### 6. Deployment / Usage
The script is designed to run directly from the terminal with customizable arguments.

#### Prerequisites
You need both Python and R installed on your system.
* **Python Libraries:** `pandas`, `numpy`, `requests`, `scikit-learn`, `rpy2`
* **R Packages:** `C50`, `partykit`

#### Execution
Run the script via terminal. It will automatically download the data, train the model, print the evaluation metrics, and generate the PDF visualization.

```bash
# Run with default settings (Adult dataset, single tree for visualization)
python main.py --dataset adult --trials 1

# Run with boosting (25 trials) for higher accuracy (Note: visualization may be too complex)
python main.py --dataset adult --trials 25 --mincases 10 --cf 0.1