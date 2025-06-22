import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ────────────────────────────────
# Load trained model + encoders
# ────────────────────────────────
MODEL_PATH = "../models/salary_predictor.pkl"
LE_JOB_PATH = "../models/le_job.pkl"
LE_LOC_PATH = "../models/le_loc.pkl"
LE_EDU_PATH = "../models/le_edu.pkl"
MLB_SKILLS_PATH = "../models/mlb_skills.pkl"

model   = joblib.load(MODEL_PATH)
le_job  = joblib.load(LE_JOB_PATH)
le_loc  = joblib.load(LE_LOC_PATH)
le_edu  = joblib.load(LE_EDU_PATH)
mlb     = joblib.load(MLB_SKILLS_PATH)

# ────────────────────────────────
# Helper: preprocess a single input
# ────────────────────────────────
def preprocess_input(job_title, skills, location, education_level, years_experience):
    """
    Turn raw user inputs into the one-row feature DataFrame expected by the model.
    """
    # Encode job title
    job_enc = le_job.transform([job_title])[0] if job_title in le_job.classes_ else -1

    # Encode location
    loc_enc = le_loc.transform([location])[0] if location in le_loc.classes_ else -1

    # Encode education level
    edu_enc = le_edu.transform([education_level])[0] if education_level in le_edu.classes_ else -1

    # One-hot encode skills
    skill_vec = np.zeros(len(mlb.classes_), dtype=int)
    for s in skills:
        if s in mlb.classes_:
            skill_vec[np.where(mlb.classes_ == s)[0][0]] = 1

    # Assemble feature vector
    features = np.concatenate([skill_vec,
                               [job_enc, loc_enc, edu_enc, years_experience]])

    col_names = list(mlb.classes_) + [
        "job_title_enc",
        "location_enc",
        "education_level_enc",
        "years_experience",
    ]
    return pd.DataFrame([features], columns=col_names)


# ────────────────────────────────
# Streamlit UI
# ────────────────────────────────
st.title(" Sri Lankan Tech Salary Predictor")

# ----- User inputs -----
job_title = st.selectbox(
    "Job Title",
    [
        "Software Engineer", "Data Scientist", "DevOps Engineer",
        "Business Analyst", "Frontend Developer", "Backend Developer",
        "Fullstack Developer", "QA Engineer", "Product Manager",
    ],
)

skills_list = [
    "Python", "Machine Learning", "SQL", "JavaScript", "React", "Docker",
    "Kubernetes", "AWS", "Azure", "Linux", "TensorFlow", "Pandas",
    "Tableau", "Excel", "Git",
]
skills = st.multiselect("Skills", skills_list)

location = st.selectbox("Location", ["Colombo", "Kandy", "Galle", "Jaffna", "Remote"])

education_level = st.selectbox(
    "Education Level",
    ["High School", "Diploma", "Bachelor's Degree", "Master's Degree", "PhD"],
)

years_experience = st.slider("Years of Experience", 0, 30, 3)

# ----- Prediction -----
if st.button("Predict Salary"):
    X_input = preprocess_input(
        job_title, skills, location, education_level, years_experience
    )
    salary_lkr = model.predict(X_input)[0]
    st.success(f"Estimated Monthly Salary: **Rs. {salary_lkr:,.0f}**")
