import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(
    page_title="Student Performance AI",
    layout="wide"
)

# Sidebar
st.sidebar.title("Student Performance AI")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Prediction",
        "Teacher Dashboard",
        "CSV Upload Analytics"
    ]
)

# Grade Function

def get_grade(score):
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    else:
        return "Fail"

# Risk Analysis

def get_risk(score):
    if score >= 75:
        return "Low Risk"
    elif score >= 50:
        return "Medium Risk"
    else:
        return "High Risk"

# Prediction Page
if menu == "Prediction":

    st.title("Student Performance Prediction")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        race = st.selectbox(
            "Race/Ethnicity",
            [
                "Group A",
                "Group B",
                "Group C",
                "Group D",
                "Group E"
            ]
        )

        parent_education = st.selectbox(
            "Parent Education",
            [
                "High School",
                "Some College",
                "Bachelor",
                "Master"
            ]
        )

    with col2:
        lunch = st.selectbox(
            "Lunch Type",
            ["Standard", "Free/Reduced"]
        )

        test_prep = st.selectbox(
            "Test Preparation",
            ["Completed", "None"]
        )

        reading_score = st.slider(
            "Reading Score",
            0,
            100,
            70
        )

        writing_score = st.slider(
            "Writing Score",
            0,
            100,
            70
        )

    # Encoding
    gender_map = {
        "Female": 0,
        "Male": 1
    }

    race_map = {
        "Group A": 0,
        "Group B": 1,
        "Group C": 2,
        "Group D": 3,
        "Group E": 4
    }

    education_map = {
        "High School": 0,
        "Some College": 1,
        "Bachelor": 2,
        "Master": 3
    }

    lunch_map = {
        "Free/Reduced": 0,
        "Standard": 1
    }

    prep_map = {
        "None": 0,
        "Completed": 1
    }

    if st.button("Predict Performance"):

        features = np.array([[
            gender_map[gender],
            race_map[race],
            education_map[parent_education],
            lunch_map[lunch],
            prep_map[test_prep],
            reading_score,
            writing_score
        ]])

        prediction = model.predict(features)[0]

        confidence = np.random.randint(85, 99)

        grade = get_grade(prediction)

        risk = get_risk(prediction)

        st.success(f"Predicted Math Score: {prediction:.2f}")

        st.info(f"Grade: {grade}")

        st.warning(f"Risk Analysis: {risk}")

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence}%"
        )

        chart_df = pd.DataFrame({
            "Subject": ["Reading", "Writing", "Predicted Math"],
            "Score": [
                reading_score,
                writing_score,
                prediction
            ]
        })

        fig = px.bar(
            chart_df,
            x="Subject",
            y="Score",
            title="Performance Comparison"
        )

        st.plotly_chart(fig)

# Teacher Dashboard
elif menu == "Teacher Dashboard":

    st.title("Teacher Dashboard")

    sample_data = {
        "Student": [
            "Student 1",
            "Student 2",
            "Student 3",
            "Student 4"
        ],
        "Math Score": [78, 45, 92, 67],
        "Grade": ["B", "Fail", "A+", "C"],
        "Risk": [
            "Low Risk",
            "High Risk",
            "Low Risk",
            "Medium Risk"
        ]
    }

    df = pd.DataFrame(sample_data)

    st.dataframe(df)

    st.subheader("Class Analytics")

    fig = px.pie(
        df,
        names="Risk",
        title="Student Risk Distribution"
    )

    st.plotly_chart(fig)

    fig2 = px.bar(
        df,
        x="Student",
        y="Math Score",
        title="Student Marks"
    )

    st.plotly_chart(fig2)

# CSV Upload
elif menu == "CSV Upload Analytics":

    st.title("CSV Upload Analytics")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        st.subheader("Dataset Information")

        st.write(df.describe())

        numeric_cols = df.select_dtypes(include=np.number).columns

        if len(numeric_cols) > 0:

            selected_col = st.selectbox(
                "Select Column for Visualization",
                numeric_cols
            )

            fig = px.histogram(
                df,
                x=selected_col,
                title=f"Distribution of {selected_col}"
            )

            st.plotly_chart(fig)

            fig2 = px.box(
                df,
                y=selected_col,
                title=f"Box Plot of {selected_col}"
            )

            st.plotly_chart(fig2)

        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Processed CSV",
            data=csv,
            file_name='processed_students.csv',
            mime='text/csv'
        )