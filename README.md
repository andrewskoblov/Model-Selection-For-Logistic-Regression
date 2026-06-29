# Model Selection for Logistic Regression

This project practices the **evaluation and deployment** phases of the machine learning life cycle, using logistic regression to predict whether an Airbnb host is a "superhost."

## Problem

Binary classification on the Airbnb `listings` dataset. The label is `host_is_superhost` (True/False); all other columns are features. The data is already preprocessed (one-hot encoded, scaled, missing values imputed).

## What the notebook does

1. Load the dataset and create labeled examples (`X`, `y`)
2. Split into training and test sets (90/10)
3. Train a baseline logistic regression model using the default hyperparameter `C`
4. Run a grid search with 5-fold cross-validation to find the optimal `C`
5. Train and evaluate the tuned model
6. Compare the default and tuned models using:
   - Confusion matrices
   - Precision-recall curves
   - ROC curves and AUC
7. Perform feature selection with `SelectKBest`
8. Save the best model with `pickle` for future use

## Files

- `ModelSelectionForLogisticRegression.ipynb` — the completed notebook
- `model_best.pkl` — the trained best-performing model, pickled
- `airbnbData_train.csv` — the training dataset

## Tools

Python, scikit-learn, pandas, NumPy, matplotlib, seaborn
