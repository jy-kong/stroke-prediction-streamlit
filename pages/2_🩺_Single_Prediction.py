# +
import streamlit as st
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import lines

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
# -
st.set_page_config(layout="centered", page_title='Stroke Prediction Oracle', page_icon='🔮')


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# +
#local_css("style/style.css")
# -

if 'result' not in st.session_state:
    st.session_state.result = 100

if 'prediction_outcome' not in st.session_state:
    st.session_state.prediction_outcome = ''

if 'prediction_probability' not in st.session_state:
    st.session_state.prediction_probability = 0

st.title("Single Prediction")

col1, col2 = st.columns(2, gap='small')

with st.form('Single Prediction', clear_on_submit=True):
    with col1:
        # Gender: Female, Male, Other (radio choice)
        gender = st.radio('Select Gender:', options=['Female', 'Male', 'Other'])

        # Age: integer (number input) allow 1 to 100 only
        age = st.number_input('Enter Age:', min_value=1, max_value=100, step=1)

        # History of Hypertension: Yes, No (radio choice)
        hypertension = st.radio('History of Hypertension:', options=['Yes', 'No'])

        # History of Heart Disease: Yes, No (radio choice)
        heart_disease = st.radio('History of Heart Disease:', options=['Yes', 'No'])

        # Ever Married: Yes, No (radio choice)
        ever_married = st.radio('Ever Married:', options=['Yes', 'No'])

        # Work Type: select box
        work_type = st.selectbox('Work Type:', 
                                  options=['Private job', 'Self-employed', 'Governmental job', 
                                           'Children', 'Never worked'])

    with col2:
        # Residence Type: Urban, Rural (radio choice)
        residence_type = st.radio('Residence Type:', options=['Urban', 'Rural'])

        # Body Mass Index (BMI): numeric input (positive float only)
        bmi = st.number_input('Enter Body Mass Index (BMI):', min_value=0.0, step=0.1, format="%.1f")

        # Average Glucose Level (mg/dL): numeric input (positive float only)
        avg_glucose_level = st.number_input('Enter Average Glucose Level (mg/dL):', min_value=0.0, step=0.1, format="%.1f")

        # Smoking Status: select box
        smoking_status = st.selectbox('Smoking Status:', 
                                      options=['Never smoked', 'Formerly smoked', 'Smokes', 'Unknown'])

    # Submit button
    submitted = st.form_submit_button("Predict")


    # Add input data into a numpy array when the form is submitted
    if submitted:
        # Map hypertension and heart disease responses to numeric values
        hypertension = 1 if hypertension == 'Yes' else 0
        heart_disease = 1 if heart_disease == 'Yes' else 0

        # Map work type responses to corresponding categories
        work_type_mapping = {
            'Private job': 'Private',
            'Self-employed': 'Self-employed',
            'Governmental job': 'Govt_job',
            'Children': 'children',
            'Never worked': 'Never_worked'
        }
        work_type = work_type_mapping.get(work_type, work_type)

        # Map smoking status responses to corresponding categories
        smoking_status_mapping = {
            'Never smoked': 'never smoked',
            'Formerly smoked': 'formerly smoked',
            'Smokes': 'smokes',
            'Unknown': 'Unknown'
        }
        smoking_status = smoking_status_mapping.get(smoking_status, smoking_status)

        # Create a dictionary of input data
        input_data = {
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'ever_married': [ever_married],
            'work_type': [work_type],
            'residence_type': [residence_type],
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi],
            'smoking_status': [smoking_status]
        }


        # Convert the dictionary into a DataFrame
        df = pd.DataFrame(input_data)

        # Load the pre-trained RandomForest model
        with open('classifier_rf_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        # Load the preprocessor (ColumnTransformer and LabelEncoders)
        with open('preprocessor.pkl', 'rb') as f:
            ct, le1, le2 = pickle.load(f)

        # Load the StandardScaler
        with open('scaler.pkl', 'rb') as f:
            sc = pickle.load(f)

        # Preprocess the input data
        # Apply the ColumnTransformer
        sample_data_transformed = ct.transform(df)

        # Apply LabelEncoder transformations to specific columns
        sample_data_transformed[:, 15] = le1.transform(sample_data_transformed[:, 15])  # 'ever_married' column
        sample_data_transformed[:, 16] = le2.transform(sample_data_transformed[:, 16])  # 'residence_type' column

        # Standardize the data
        sample_data_scaled = sc.transform(sample_data_transformed)

        # Make predictions
        prediction = loaded_model.predict(sample_data_scaled)
        st.session_state.prediction_probability = loaded_model.predict_proba(sample_data_scaled)

        # Display the results
        if prediction == 0:
            st.session_state.result = 0
            st.session_state.prediction_outcome = "No Stroke"
        else:
            st.session_state.result = 1
            st.session_state.prediction_outcome = "Stroke"

image_path = ''
col3, col4 = st.columns(2)

with col3:
    if st.session_state.result == 0:
        st.subheader("Prediction Result")
        image_path = 'healthy.png'
        st.image(image_path, width=200)
    elif st.session_state.result == 1:
        st.subheader("Prediction Result")
        image_path = 'stroke.png'
        st.image(image_path, width=200)

with col4:
    if st.session_state.result == 0:
        st.header(f':green[{st.session_state.prediction_outcome}]')
        st.subheader(f"You have {st.session_state.prediction_probability[0, 0] * 100:.0f}% probability of not having a stroke.")
    elif st.session_state.result == 1:
        st.header(f':red[{st.session_state.prediction_outcome}]')
        st.subheader(f"You have {st.session_state.prediction_probability[0, 1] * 100:.0f}% probability of having a stroke.")

if st.session_state.result == 0 or st.session_state.result == 1:
    # Trigger for the pop-up
    if st.button("Prediction Insights!"):
        with st.expander("Machine Learning \"Black Box\"", expanded=True):
            st.write("Let's dive into how the Random Forest (RF) model made predictions by examining the feature importance of each variable!")


            # Load the pre-trained RandomForest model
            with open('classifier_rf_model.pkl', 'rb') as f:
                loaded_model = pickle.load(f)

            # Define feature names corresponding to the columns in the training dataset
            feature_names = [
                'gender_female', 'gender_male', 'gender_other',
                'work_type_govt_job', 'work_type_never_worked', 'work_type_private',
                'work_type_self_employed', 'work_type_children', 'smoking_status_unknown',
                'smoking_status_formerly_smoked', 'smoking_status_never_smoked', 'smoking_status_smokes',
                'age', 'hypertension', 'heart_disease', 'ever_married',
                'residence_type', 'avg_glucose_level', 'bmi'
            ]

            # Create DataFrame with feature names and importances
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': loaded_model.feature_importances_
            })

            # Sort features by importance
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

            # Visualization setup
            background_color = "#fbfbfb"
            fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor=background_color)

            # Color mapping: Highlight top 3 features
            color_map = ['lightgray' for _ in range(len(feature_importance_df))]
            color_map[0] = color_map[1] = color_map[2] = '#0f4c81'  # Highlight top 3 features

            # Bar plot of feature importance
            sns.barplot(
                data=feature_importance_df,
                x='Importance',
                y='Feature',
                ax=ax,
                palette=color_map
            )

            # Adjust plot aesthetics
            ax.set_facecolor(background_color)
            for spine in ['top', 'left', 'right']:
                ax.spines[spine].set_visible(False)

            fig.text(
                0.12, 0.92,
                "Feature Importance: Random Forest Stroke Prediction",
                fontsize=18, fontweight='bold', fontfamily='serif'
            )

            fig.text(
                0.95, 0.92,
                "Insight",
                fontsize=18, fontweight='bold', fontfamily='serif'
            )

            fig.text(
                0.85, 0.315,
                '''
                It is always interesting to view what features
                a predictive model utilizes the most, that is,
                what features are the most important.
                This not only helps understand how the model
                works, but importantly can help us to explain
                the model results.

                In this case, we see that Age, Average Glucose Level,
                and BMI are the most important factors for our model.

                One also notices just how important Age is for our model,
                it is by far the most significant variable.

                It is also interesting that Work Type is more salient
                than Gender - this is a surprise.

                Having a history of Heart Disease and Hypertension
                are also low in the importance ranking which again
                is very surprising.
                ''',
                fontsize=14, fontweight='light', fontfamily='serif'
            )

            ax.tick_params(axis='both', which='both', length=0)

            # Add vertical line
            l1 = lines.Line2D(
                [0.98, 0.98],
                [0, 1],
                transform=fig.transFigure,
                figure=fig,
                color='black',
                lw=0.2
            )
            fig.lines.extend([l1])

            plt.xlabel("", fontsize=12, fontweight='light', fontfamily='serif', loc='left')
            plt.ylabel("", fontsize=12, fontweight='light', fontfamily='serif')

            # Integrate with Streamlit
            st.title("Feature Importance Visualization")
            st.pyplot(fig)


            st.button("Close")  # Add a button for users to "close" it
