import streamlit as st
import pandas as pd
import joblib
from data_preprocessing import encode, normalize

# Load the saved best model
@st.cache_resource
def load_best_model():
    return joblib.load('./model/best_model.pkl')

model = load_best_model()

# Streamlit app title
st.title("Employee Attrition Prediction")

# Demographic Information Section
st.header("Demographic Information")
age = st.number_input("Age", min_value=18, max_value=60, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

# Job Details Section
st.header("Job Details")
job_role = st.selectbox("Job Role", [
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative", "Manager",
    "Sales Representative", "Research Director", "Human Resources"
])
department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)

# Work-Life Balance and Satisfaction Section
st.header("Work-Life Balance & Satisfaction")
work_life_balance = st.selectbox("Work Life Balance (1: Low, 4: High)", [1, 2, 3, 4])
job_satisfaction = st.selectbox("Job Satisfaction (1: Low, 4: High)", [1, 2, 3, 4])
environment_satisfaction = st.selectbox("Environment Satisfaction (1: Low, 4: High)", [1, 2, 3, 4])

# Other Job-Related Factors
st.header("Other Job-Related Factors")
num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=10, value=1)
years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=20, value=2)
years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=1)
years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, max_value=20, value=3)

# Collect all input data for prediction
input_data = {
    "Age": age,
    "Gender": gender,
    "MaritalStatus": marital_status,
    "JobRole": job_role,
    "Department": department,
    "JobLevel": job_level,
    "MonthlyIncome": monthly_income,
    "WorkLifeBalance": work_life_balance,
    "JobSatisfaction": job_satisfaction,
    "EnvironmentSatisfaction": environment_satisfaction,
    "NumCompaniesWorked": num_companies_worked,
    "YearsAtCompany": years_at_company,
    "YearsInCurrentRole": years_in_current_role,
    "YearsSinceLastPromotion": years_since_last_promotion,
    "YearsWithCurrManager": years_with_curr_manager
}

# When the Predict button is clicked
if st.button("Predict"):
    # Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Preprocess the input data
    input_df = encode(input_df)
    input_df = normalize(input_df)
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Display prediction
    st.subheader("Prediction Result")
    st.write("Attrition Prediction:", "Yes" if prediction[0] == 1 else "No")
