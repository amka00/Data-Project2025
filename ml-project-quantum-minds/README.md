# Project Overview

This project focuses on predicting the risk of developing coronary heart disease (MICHD) using data from the Behavioral Risk Factor Surveillance System (BRFSS). The work combines classical statistical learning techniques with modern machine learning models to analyze a large, real-world medical dataset.

---

## Project Objectives

- Perform data preprocessing and cleaning, including handling missing values and feature scaling  
- Apply feature selection techniques to improve model performance  
- Implement and compare a range of machine learning models, from linear baselines to non-linear methods  
- Optimize models using appropriate evaluation metrics  
- Generate predictions for submission to an external evaluation platform  

---

## Dataset

The project uses three CSV datasets:

- `x_train.csv`: Training feature set  
- `y_train.csv`: Binary labels indicating MICHD presence  
- `x_test.csv`: Test feature set for prediction submission  

The dataset contains demographic, behavioral, and lifestyle features relevant to coronary heart disease risk.

---

## Methodology

### Data Preprocessing

- Removal of features with a high proportion of missing values  
- Mean imputation for remaining missing data  
- Feature standardization to zero mean and unit variance  
- Correlation-based feature selection  

### Model Implementations

The following models are implemented and evaluated:

**Linear and Statistical Models**
- Linear Regression  
- Least Squares Regression  
- Linear Regression with Gradient Descent and Stochastic Gradient Descent  
- Ridge Regression  
- Logistic Regression  
- Regularized Logistic Regression  

**Machine Learning Models**
- Random Forest  
- XGBoost  
- Neural Network (Multilayer Perceptron)

Linear and statistical models are implemented from scratch using NumPy, while advanced machine learning models rely on established libraries such as scikit-learn and XGBoost.

---

## Model Training and Evaluation

Models are trained on balanced subsets of the data to address class imbalance. Hyperparameters are tuned using validation data. Performance is evaluated using accuracy and F1-score. The best-performing model is used to generate predictions for final submission.

---

## Results

The project compares model performance across all approaches and highlights the gains achieved through non-linear machine learning techniques. Results include evaluation metrics and visualizations illustrating model behavior and feature importance.

---

## Getting Started

Clone the repository:
```bash
git clone https://github.com/amka00/data-project2025.git
