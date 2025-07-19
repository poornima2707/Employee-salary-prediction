import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# Set page title
st.title("Employee Salary Prediction")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("adult.csv")  # Make sure to upload this file in Streamlit
    return data

data = load_data()

# Data preprocessing
def preprocess_data(df):
    # Replace '?' with NaN
    df = df.replace('?', np.nan)
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Replace '?' with 'Others'
    df['workclass'].replace({'?':'Others'}, inplace=True)
    df['occupation'].replace({'?':'Others'}, inplace=True)
    
    # Remove rare categories
    df = df[df['workclass'] != 'Without-pay']
    df = df[df['workclass'] != 'Never-worked']
    
    # Filter outliers
    df = df[(df['age'] <= 75) & (df['age'] >= 17)]
    df = df[(df['educational-num'] <= 16) & (df['educational-num'] >= 5)]
    
    return df

# Preprocess the data
data = preprocess_data(data)

# Train or load model
@st.cache_resource
def get_model():
    # Check if model exists
    if os.path.exists('salary_model.pkl'):
        model = joblib.load('salary_model.pkl')
    else:
        # Prepare data for training
        X = data.drop(['income', 'education', 'fnlwgt', 'native-country'], axis=1)
        y = data['income']
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
        
        # Train the model
        model = GradientBoostingClassifier()
        model.fit(X, y)
        
        # Save the model and encoders
        joblib.dump(model, 'salary_model.pkl')
        joblib.dump(label_encoders, 'label_encoders.pkl')
    
    return model

model = get_model()

# Load label encoders
try:
    label_encoders = joblib.load('label_encoders.pkl')
except:
    label_encoders = None

# Sidebar for user input
st.sidebar.header("Input Features")

def user_input_features():
    age = st.sidebar.slider('Age', 17, 75, 30)
    workclass = st.sidebar.selectbox('Workclass', data['workclass'].unique())
    education_num = st.sidebar.slider('Education Level (1-16)', 1, 16, 9)
    marital_status = st.sidebar.selectbox('Marital Status', data['marital-status'].unique())
    occupation = st.sidebar.selectbox('Occupation', data['occupation'].unique())
    relationship = st.sidebar.selectbox('Relationship', data['relationship'].unique())
    race = st.sidebar.selectbox('Race', data['race'].unique())
    gender = st.sidebar.selectbox('Gender', data['gender'].unique())
    capital_gain = st.sidebar.slider('Capital Gain', 0, 100000, 0)
    capital_loss = st.sidebar.slider('Capital Loss', 0, 5000, 0)
    hours_per_week = st.sidebar.slider('Hours per Week', 1, 100, 40)
    
    user_data = {
        'age': age,
        'workclass': workclass,
        'educational-num': education_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week
    }
    
    return pd.DataFrame(user_data, index=[0])

input_df = user_input_features()

# Main panel
st.subheader("User Input Features")
st.write(input_df)

# Prediction
if st.sidebar.button('Predict'):
    if label_encoders is None:
        st.error("Label encoders not found. Please retrain the model.")
    else:
        # Preprocess input data
        input_processed = input_df.copy()
        for col in label_encoders:
            if col in input_processed.columns:
                input_processed[col] = label_encoders[col].transform(input_processed[col])
        
        # Remove extra columns that aren't in the model
        input_processed = input_processed.drop(['education', 'fnlwgt', 'native-country'], axis=1, errors='ignore')
        
        # Make prediction
        prediction = model.predict(input_processed)
        prediction_proba = model.predict_proba(input_processed)
        
        st.subheader("Prediction")
        st.write(f"Income: {'>50K' if prediction[0] == 1 else '<=50K'}")
        
        st.subheader("Prediction Probability")
        st.write(f"Probability of <=50K: {prediction_proba[0][0]:.2f}")
        st.write(f"Probability of >50K: {prediction_proba[0][1]:.2f}")

# Data Exploration
if st.checkbox("Show Data Exploration"):
    st.header("Data Exploration")
    
    st.subheader("Income Distribution")
    fig, ax = plt.subplots()
    data['income'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)
    
    st.subheader("Age Distribution by Income")
    fig, ax = plt.subplots()
    for income in data['income'].unique():
        data[data['income'] == income]['age'].plot(kind='kde', ax=ax, label=income)
    ax.legend()
    st.pyplot(fig)