# 🫀 Heart Disease Prediction Project

## 📖 Summary
This project develops robust machine learning models to predict heart disease and identify the most significant clinical risk factors influencing diagnosis. Using a combined dataset from the **UCI Machine Learning Repository** (Cleveland, Hungary, Switzerland, VA Long Beach), the study applies preprocessing, exploratory analysis, feature engineering, and model development to achieve high predictive accuracy and clinical interpretability.

---

## 🎯 Objectives
- **Predict Heart Disease**: Build accurate models to classify patients with or without heart disease.
- **Identify Risk Factors**: Pinpoint key demographic and clinical variables contributing to disease risk.
- **Understand Feature Importance**: Provide interpretable insights into the relative influence of predictors.

---

## 📊 Dataset
- **Source**: [UCI Machine Learning Repository – Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Records**: 920 patients
- **Original Features**: 16 demographic and clinical variables
- **Expanded Features**: 37 after preprocessing (one-hot encoding, engineered interactions)
- **Target Variable**: Binary classification (0 = no disease, 1 = disease)
- **Class Balance**: Corrected using **SMOTE** oversampling

---

## 🔬 Methodology
### Preprocessing
- Median imputation for missing numerical values
- Categorical imputation with `"Missing"` label
- StandardScaler normalization
- One-hot encoding for categorical variables
- Feature engineering: `age_sex_interaction`, `age_squared`, `chol_squared`

### Models
- **Logistic Regression** (interpretable, linear baseline)
- **Random Forest Classifier** (ensemble, non-linear relationships)

### Hyperparameter Tuning
- GridSearchCV with 5-fold cross-validation
- ROC AUC as scoring metric

---

## 📈 Results
| Model                  | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|-------------------------|----------|-----------|--------|----------|---------|
| Logistic Regression     | 0.7990   | 0.8352    | 0.7451 | 0.7876   | 0.9041  |
| Random Forest           | 0.8088   | 0.8462    | 0.7549 | 0.7979   | 0.8971  |
| Tuned Logistic Regression | 0.8039 | 0.8298    | 0.7647 | 0.7959   | **0.9112** |
| Tuned Random Forest     | 0.8039   | 0.8298    | 0.7647 | 0.7959   | 0.9019  |

- Logistic Regression slightly outperformed Random Forest in ROC AUC.
- Both models achieved **ROC AUC > 0.90**, demonstrating excellent discriminatory power.
- Logistic Regression was selected as the preferred model due to interpretability and clinical relevance.

---

## 🔑 Key Insights
- Strong predictors: **number of major vessels (ca)**, **ST depression (oldpeak)**, **exercise-induced angina (exang)**, **dataset Switzerland**, **slope flat**
- Protective factors: Certain chest pain types, normal thallium test results
- Random Forest captured non-linear effects in **age** and **cholesterol**
- SHAP and permutation importance confirmed consistent feature relevance

---

## 🛠️ Tools & Libraries
- **Python (Google Colab)**
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Statsmodels
- SHAP

---

## 📓 Notebook
- [Project Notebook (Google Colab)](https://colab.research.google.com/drive/1OXVwVHni8EaM0DyiZqlpYxMxCtwICbrT)

---

## ▶️ Instructions to Run
1. Open the **Project** notebook link above.
2. Install required libraries:
   ```bash
   !pip install pandas numpy matplotlib seaborn scikit-learn shap statsmodels
