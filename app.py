import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Student Dropout Prediction", layout="wide")

# Title and description
st.title("Student Dropout Prediction Dashboard")
st.markdown("""
This application predicts whether a student will **Graduate**, **Dropout**, or remain **Enrolled** based on academic, demographic, and socio-economic features.
Use the sidebar to explore data, view model performance, or predict outcomes for new students.
""")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Data Exploration", "Model Performance", "Predict New Student"])

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv("data.csv", sep=";")
    # Clean data
    data = data.dropna()  # Remove rows with missing values
    data = data[data["Status"].isin(["Graduate", "Dropout", "Enrolled"])]  # Ensure valid target values
    return data

# Preprocess data for modeling
@st.cache_data
def preprocess_data(data):
    # Encode categorical target variable
    le_status = LabelEncoder()
    data["Status"] = le_status.fit_transform(data["Status"])
    
    # Separate features and target
    X = data.drop("Status", axis=1)
    y = data["Status"]
    
    # Encode categorical features
    categorical_cols = ["Marital_status", "Application_mode", "Course", "Daytime_evening_attendance", 
                        "Previous_qualification", "Nacionality", "Mothers_qualification", 
                        "Fathers_qualification", "Mothers_occupation", "Fathers_occupation", 
                        "Displaced", "Educational_special_needs", "Debtor", "Tuition_fees_up_to_date", 
                        "Gender", "Scholarship_holder", "International"]
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # Scale numerical features
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y, le_status, scaler

# Train model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define and tune Random Forest model
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Save the best model
    joblib.dump(grid_search.best_estimator_, "model.pkl")
    joblib.dump(X.columns, "feature_names.pkl")
    
    return grid_search.best_estimator_, X_train, X_test, y_train, y_test

# Load data
data = load_data()

# Data Exploration Page
if page == "Data Exploration":
    st.header("Data Exploration")
    
    # Display dataset overview
    st.subheader("Dataset Overview")
    st.write(f"Number of records: {data.shape[0]}")
    st.write(f"Number of features: {data.shape[1] - 1}")
    st.dataframe(data.head())
    
    # Plot distribution of Status
    st.subheader("Distribution of Student Status")
    fig_status = px.histogram(data, x="Status", title="Distribution of Student Status")
    st.plotly_chart(fig_status, use_container_width=True)
    
    # Plot distribution of Age_at_enrollment
    st.subheader("Distribution of Age at Enrollment")
    fig_age = px.histogram(data, x="Age_at_enrollment", nbins=30, title="Age Distribution")
    st.plotly_chart(fig_age, use_container_width=True)
    
    # Interesting insight
    st.subheader("Interesting Insight")
    dropout_rate = data["Status"].value_counts(normalize=True)["Dropout"] * 100
    st.write(f"**Insight**: {dropout_rate:.2f}% of students drop out, indicating a significant portion of the student population may need targeted interventions.")

# Model Performance Page
elif page == "Model Performance":
    st.header("Model Performance")
    
    # Preprocess data
    X, y, le_status, scaler = preprocess_data(data)
    
    # Train and evaluate model
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    y_pred = model.predict(X_test)
    
    # Display classification report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=le_status.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
    # Display confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=le_status.classes_,
        y=le_status.classes_,
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        showscale=True
    ))
    fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Feature importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    fig_fi = px.bar(feature_importance, x="Importance", y="Feature", title="Feature Importance")
    st.plotly_chart(fig_fi, use_container_width=True)

# Predict New Student Page
elif page == "Predict New Student":
    st.header("Predict Outcome for a New Student")
    
    # Load model and feature names
    model = joblib.load("model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    
    # Create input form
    st.subheader("Enter Student Details")
    input_data = {}
    with st.form("prediction_form"):
        for feature in feature_names:
            if feature in ["Marital_status", "Application_mode", "Course", "Daytime_evening_attendance", 
                          "Previous_qualification", "Nacionality", "Mothers_qualification", 
                          "Fathers_qualification", "Mothers_occupation", "Fathers_occupation", 
                          "Displaced", "Educational_special_needs", "Debtor", "Tuition_fees_up_to_date", 
                          "Gender", "Scholarship_holder", "International"]:
                input_data[feature] = st.selectbox(f"{feature}", sorted(data[feature].unique()))
            else:
                input_data[feature] = st.number_input(f"{feature}", value=float(data[feature].mean()))
        
        submitted = st.form_submit_button("Predict")
    
    if submitted:
        # Prepare input data
        input_df = pd.DataFrame([input_data])
        for col in ["Marital_status", "Application_mode", "Course", "Daytime_evening_attendance", 
                    "Previous_qualification", "Nacionality", "Mothers_qualification", 
                    "Fathers_qualification", "Mothers_occupation", "Fathers_occupation", 
                    "Displaced", "Educational_special_needs", "Debtor", "Tuition_fees_up_to_date", 
                    "Gender", "Scholarship_holder", "International"]:
            le = LabelEncoder()
            le.fit(data[col])
            input_df[col] = le.transform(input_df[col])
        
        # Scale numerical features
        numerical_cols = input_df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        scaler.fit(data[numerical_cols])
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        
        # Predict
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        # Display prediction
        st.subheader("Prediction Result")
        predicted_status = le_status.inverse_transform(prediction)[0]
        st.write(f"**Predicted Status**: {predicted_status}")
        st.write("**Prediction Probabilities**:")
        proba_df = pd.DataFrame({
            "Status": le_status.classes_,
            "Probability": prediction_proba[0]
        })
        st.dataframe(proba_df)

# Footer
st.markdown("""
---
Built with Streamlit and deployed on Streamlit Community Cloud.
Dataset source: Student academic performance data.
""")