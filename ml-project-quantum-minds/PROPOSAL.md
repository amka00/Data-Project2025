# Project Proposal

## Title  
Predicting Coronary Heart Disease Risk Using Machine Learning on BRFSS Health Data

## Category  
Data Science / Medical Machine Learning

---

## Problem Statement and Motivation  

Cardiovascular diseases (CVDs) are among the leading causes of mortality worldwide. Coronary heart disease (MICHD) is particularly prevalent and poses significant challenges for healthcare systems. Early identification of high-risk individuals is crucial for prevention and targeted intervention.

This project aims to predict the likelihood of coronary heart disease using health-related data from the Behavioral Risk Factor Surveillance System (BRFSS). By leveraging both classical statistical learning methods and modern machine learning models, the project evaluates how different modeling approaches perform on a large, real-world medical dataset.

---

## Dataset Description  

The BRFSS dataset is a large-scale public health survey conducted annually in the United States, containing demographic, behavioral, and lifestyle information for over 300,000 individuals. Respondents are labeled as MICHD-positive if they report a diagnosis of coronary heart disease, angina, or myocardial infarction.

The project relies on three datasets:

- `x_train.csv`: training features  
- `y_train.csv`: binary labels indicating MICHD presence  
- `x_test.csv`: test features for prediction submission  

Additional information about the dataset is available at:  
https://www.cdc.gov/brfss/annual_data/annual_2015.html

---

## Planned Approach and Methodology  

### Data Preprocessing  
The data will be cleaned by removing features with excessive missing values and imputing remaining missing entries using feature means. All features will be standardized, and feature selection will be performed using correlation-based methods to reduce dimensionality and improve model performance.

### Models  
The project includes a comparison of models with increasing complexity:

**Baseline and Linear Models**
- Linear Regression  
- Least Squares Regression  
- Ridge Regression  
- Logistic Regression  
- Regularized Logistic Regression  

**Machine Learning Models**
- Random Forest  
- XGBoost  
- Neural Network (Multilayer Perceptron)

Classical models will be implemented from scratch using NumPy, while more advanced machine learning models will rely on standard Python libraries to capture non-linear relationships and interactions between variables.

### Evaluation  
Models will be evaluated using accuracy and F1-score, with particular attention to class imbalance. Hyperparameters will be tuned using validation data, and final predictions will be generated for submission to an external evaluation platform.

---

## Expected Challenges  

Challenges include class imbalance, noisy self-reported data, and correlated features. These issues will be mitigated through regularization, balanced sampling strategies, and careful feature selection.

---

## Success Criteria  

The project will be considered successful if the machine learning models outperform simple baseline approaches, achieve strong predictive performance, and provide meaningful insights into the factors associated with coronary heart disease risk.
