# 🧠 Employee Salary Prediction using Machine Learning

This project focuses on predicting whether an individual earns more than $50K per year using machine learning techniques on census income data. It involves data cleaning, feature engineering, model training, and deployment through an interactive web app.

---

## 📌 Project Overview

The primary goal is to build a classification model that can determine an employee's income bracket (`<=50K` or `>50K`) based on features like age, education, occupation, and work hours. The project walks through the complete data science workflow — from raw data to deployment.

---

## 📊 Dataset

- **Source**: UCI Machine Learning Repository – [Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **Records**: 48,842 instances
- **Attributes**: Demographic and work-related features
- **Target**: Binary income classification (`<=50K` or `>50K`)

---

## 🔧 Key Steps & Techniques

- **Data Cleaning**: Handled missing values (`?`) by replacing with `"Others"`, removed invalid rows.
- **Outlier Removal**: Filtered out extreme values for `age`, `educational-num`, and `capital-gain`.
- **Label Encoding**: Converted categorical features into numerical form using `LabelEncoder`.
- **Feature Selection**: Dropped redundant columns like `education` in favor of `educational-num`.
- **Model Building**: Applied multiple classifiers using scikit-learn.

---

## 🤖 Models Trained

| Model                | Accuracy |
|---------------------|----------|
| Logistic Regression | 79.3%    |
| Random Forest       | 85.2%    |
| KNN                 | 77.0%    |
| SVM                 | 78.8%    |
| Gradient Boosting   | **85.7%** ✅ |

✅ **Gradient Boosting** was selected as the final model.

---

## 🚀 Deployment (Streamlit)

A simple web application was created using **Streamlit**, enabling:
- Real-time prediction based on manual inputs
- Batch prediction from uploaded CSV files
- Downloading prediction results

The best-performing model was saved using `joblib` as `best_model.pkl`.

---

## 📂 Files Included

- `employee-salary-prediction.ipynb` – Jupyter Notebook with complete code
- `app.py` – Streamlit application script
- `best_model.pkl` – Trained Gradient Boosting model
- `README.md` – Project overview

---

## ▶️ To Run Locally

1. Clone this repository  
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
