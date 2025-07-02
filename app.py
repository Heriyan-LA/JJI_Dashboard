import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import plotly.express as px
import joblib

# Set page configuration
st.set_page_config(page_title="Simple Student Dropout Prediction", layout="wide")

# Title
st.title("Student Dropout Prediction")
st.markdown("Predict whether a student will Graduate, Dropout, or remain Enrolled.")

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv("data.csv", sep=";")
    # Select key features
    columns = ["Age_at_enrollment", "Admission_grade", "Curricular_units_1st_sem_grade", 
               "Curricular_units_2nd_sem_grade", "Gender", "Scholarship_holder", "Status"]
    data = data[columns].dropna()
    data = data[data["Status"].isin(["Graduate", "Dropout", "Enrolled"])]
    return data

# Preprocess data for modeling
@st.cache_data
def preprocess_data(data):
    # Encode categorical variables
    le_status = LabelEncoder()
    data["Status"] = le_status.fit_transform(data["Status"])
    le_gender = LabelEncoder()
    data["Gender"] = le_gender.fit_transform(data["Gender"])
    le_scholarship = LabelEncoder()
    data["Scholarship_holder"] = le_scholarship.fit_transform(data["Scholarship_holder"])
    
    # Features and target
    X = data.drop("Status", axis=1)
    y = data["Status"]
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ["Age_at_enrollment", "Admission_grade", "Curricular_units_1st_sem_grade", 
                      "Curricular_units_2nd_sem_grade"]
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y, le_status, scaler, le_gender, le_scholarship

# Train model
@st.cache_resource精确

System: I notice the `app.py` code was cut off. Below, I’ll complete the simplified implementation for the Streamlit app, ensuring it’s concise, functional, and ready for deployment on Streamlit Community Cloud. The app will include data exploration, model evaluation, and a prediction interface using the provided `data.csv` dataset with a reduced feature set. I’ll also provide the necessary `requirements.txt` file and deployment instructions.

---

### Completed Simplified Implementation

<xaiArtifact artifact_id="2b718ce3-54b2-4501-a9dc-c25ec7a345a1" artifact_version_id="e44beb4a-84e6-4572-812c-41c93ee59e0b" title="app.py" contentType="text/python">
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Set page configuration
st.set_page_config(page_title="Simple Student Dropout Prediction", layout="wide")

# Title
st.title("Student Dropout Prediction")
st.markdown("Predict whether a student will **Graduate**, **Dropout**, or remain **Enrolled** based on key features.")

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv("data.csv", sep=";")
    # Select key features
    columns = ["Age_at_enrollment", "Admission_grade", "Curricular_units_1st_sem_grade", 
               "Curricular_units_2nd_sem_grade", "Gender", "Scholarship_holder", "Status"]
    data = data[columns].dropna()
    data = data[data["Status"].isin(["Graduate", "Dropout", "Enrolled"])]
    return data

# Preprocess data for modeling
@st.cache_data
def preprocess_data(data):
    # Encode categorical variables
    le_status = LabelEncoder()
    data["Status"] = le_status.fit_transform(data["Status"])
    le_gender = LabelEncoder()
    data["Gender"] = le_gender.fit_transform(data["Gender"])
    le_scholarship = LabelEncoder()
    data["Scholarship_holder"] = le_scholarship.fit_transform(data["Scholarship_holder"])
    
    # Features and target
    X = data.drop("Status", axis=1)
    y = data["Status"]
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ["Age_at_enrollment", "Admission_grade", "Curricular_units_1st_sem_grade", 
                      "Curricular_units_2nd_sem_grade"]
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y, le_status, scaler, le_gender, le_scholarship

# Train model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and encoders
    joblib.dump(model, "model.pkl")
    joblib.dump(X.columns, "feature_names.pkl")
    return model, X_test, y_test

# Load data
data = load_data()
X, y, le_status, scaler, le_gender, le_scholarship = preprocess_data(data)
model, X_test, y_test = train_model(X, y)

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Data Exploration", "Model Evaluation", "Predict"])

# Data Exploration Page
if page == "Data Exploration":
    st.header("Data Exploration")
    st.write(f"Dataset size: {data.shape[0]} records, {data.shape[1] - 1} features")
    st.dataframe(data.head())
    
    # Status distribution
    st.subheader("Distribution of Student Status")
    fig = px.histogram(data, x="Status", title="Student Status Distribution")
    st.plotly_chart(fig, use_container_width=True)

# Model Evaluation Page
elif page == "Model Evaluation":
    st.header("Model Evaluation")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Model Accuracy**: {accuracy:.2f}")
    
    # Confusion matrix
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

# Prediction Page
elif page == "Predict":
    st.header("Predict Student Outcome")
    with st.form("prediction_form"):
        age = st.number_input("Age at Enrollment", min_value=15, max_value=70, value=20)
        admission_grade = st.number_input("Admission Grade", min_value=0.0, max_value=200.0, value=130.0)
        sem1_grade = st.number_input("1st Semester Grade", min_value=0.0, max_value=20.0, value=12.0)
        sem2_grade = st.number_input("2nd Semester Grade", min_value=0.0, max_value=20.0, value=12.0)
        gender = st.selectbox("Gender", options=["Male", "Female"])
        scholarship = st.selectbox("Scholarship Holder", options=["No", "Yes"])
        submit = st.form_submit_button("Predict")
    
    if submit:
        # Prepare input
        input_data = pd.DataFrame({
            "Age_at_enrollment": [age],
            "Admission_grade": [admission_grade],
            "Curricular_units_1st_sem_grade": [sem1_grade],
            "Curricular_units_2nd_sem_grade": [sem2_grade],
            "Gender": [1 if gender == "Male" else 0],
            "Scholarship_holder": [1 if scholarship == "Yes" else 0]
        })
        # Scale numerical features
        numerical_cols = ["Age_at_enrollment", "Admission_grade", "Curricular_units_1st_sem_grade", 
                          "Curricular_units_2nd_sem_grade"]
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
        
        # Predict
        prediction = model.predict(input_data)
        predicted_status = le_status.inverse_transform(prediction)[0]
        st.write(f"**Predicted Status**: {predicted_status}")

# Footer
st.markdown("---\nBuilt with Streamlit for simple deployment.")