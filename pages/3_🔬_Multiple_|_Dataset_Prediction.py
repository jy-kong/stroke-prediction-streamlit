import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

st.set_page_config(
    page_title='Stroke Prediction Oracle',
    layout='centered',
    page_icon='🔮'
)


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# +
#local_css("style/style.css")
# -

data = None;

st.title('Multiple/Dataset Prediction')

st.markdown(
    """
    #### Important Note
    Before uploading your CSV file, ensure it contains the following column headers in the correct order with values in the accepted format as described below. You can also download the CSV template below to input your stroke patients' data.

    | **Column Name**         | **Accepted Values**                                       | **Values Description**                                                                                   |
    |--------------------------|----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
    | Gender                  | **Male, Female, Other**                                   | Gender of the individual.                                                                                 |
    | Age                     | Integer values ranging from 1 to 100 (e.g., **23, 67**)   | Age of the individual.                                                                                   |
    | Hypertension            | **0, 1**                                                 | Health-related parameter indicating whether the person has hypertension. **0** for no hypertension, **1** for hypertension. |
    | Heart Disease           | **0, 1**                                                 | Health-related parameter indicating whether the person has heart disease. **0** for no heart disease, **1** for heart disease. |
    | Ever Married            | **Yes, No**                                              | Personal information indicating whether the person is married or not. **Yes** for married, **No** for not married. |
    | Work Type               | **Private, Self-employed, children, Govt_job, Never_worked** | Nature of the workplace. **Private** for private jobs, **Self-employed** for independent workers, **children** for minors, **Govt_job** for government employees, **Never_worked** for no work experience. |
    | Residence Type          | **Urban, Rural**                                         | Type of residence of the individual. **Urban** for city dwellers, **Rural** for countryside residents.     |
    | Average Glucose Level   | Positive real numbers up to 1 decimal place (e.g., **72.4, 120.3**) | Average glucose level in blood for the individual in units of **mg/dL**.                                  |
    | BMI                     | Positive real numbers up to 1 decimal place (e.g., **22.7, 36.9**) | Body mass index of the individual, calculated using the formula: **weight (kg) / height² (m²)**.          |
    | Smoking Status          | **never smoked, formerly smoked, smokes, Unknown**       | Habitual information indicating the current smoking status of the individual. **Unknown** if unspecified. |
    """
)

def convert_df(df):
    return df.to_csv(index=False)


template = pd.DataFrame(columns=[
    'Gender', 
    'Age', 
    'Hypertension', 
    'Heart Disease', 
    'Ever Married', 
    'Work Type', 
    'Residence Type', 
    'Average Glucose Level', 
    'BMI', 
    'Smoking Status'
])
template = convert_df(template)

st.download_button(
            label='Download CSV template',
            data=template,
            file_name='Template.csv',
            mime='text/csv'
        )

# +
def predict(data):
    if data is None:
        st.error("Please upload a CSV file before prediction.")
    else:
        # Check for missing columns or empty values
        required_columns = ['Gender', 'Age', 'Hypertension', 'Heart Disease', 'Ever Married', 
                            'Work Type', 'Residence Type', 'Average Glucose Level', 'BMI', 'Smoking Status']
        missing_columns = [col for col in required_columns if col not in data.columns]
        empty_columns = [col for col in data.columns if data[col].isnull().any()]

        if missing_columns:
            st.error(f"The uploaded data is missing required column(s): {', '.join(missing_columns)}.")
            return
        if empty_columns:
            st.error(f"The uploaded data contains empty values in column(s): {', '.join(empty_columns)}.")
            return

        st.markdown("## Prediction Results Preview")

        data.columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
                        'work_type', 'residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']

        # Load the pre-trained RandomForest model
        with open('classifier_rf_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        # Load the preprocessor (ColumnTransformer and LabelEncoders)
        with open('preprocessor.pkl', 'rb') as f:
            ct, le1, le2 = pickle.load(f)

        # Load the StandardScaler
        with open('scaler.pkl', 'rb') as f:
            sc = pickle.load(f)

#         # Map string values to required encodings
#         data['hypertension'] = data['hypertension'].map({'Yes': 1, 'No': 0})
#         data['heart_disease'] = data['heart_disease'].map({'Yes': 1, 'No': 0})
#         data['ever_married'] = data['ever_married'].map({'Yes': 'Yes', 'No': 'No'})

#         # Work type mapping
#         work_type_mapping = {
#             'Private job': 'Private', 
#             'Self-employed': 'Self-employed', 
#             'Governmental job': 'Govt_job', 
#             'Children': 'children', 
#             'Never worked': 'Never_worked'
#         }
#         data['work_type'] = data['work_type'].map(work_type_mapping)

#         # Smoking status mapping
#         smoking_status_mapping = {
#             'Never smoked': 'never smoked', 
#             'Formerly smoked': 'formerly smoked', 
#             'Smokes': 'smokes', 
#             'Unknown': 'Unknown'
#         }
#         data['smoking_status'] = data['smoking_status'].map(smoking_status_mapping)

        # Preprocess the data
        # Apply the ColumnTransformer
        input_data = data.copy(deep=True)
        sample_data_transformed = ct.transform(input_data)

        # Apply LabelEncoder transformations to specific columns
        sample_data_transformed[:, 15] = le1.transform(sample_data_transformed[:, 15])  # 'ever_married' column
        sample_data_transformed[:, 16] = le2.transform(sample_data_transformed[:, 16])  # 'residence_type' column

        # Standardize the data
        sample_data_scaled = sc.transform(sample_data_transformed)

        # Perform batch predictions
        predictions = loaded_model.predict(sample_data_scaled)
        probabilities = loaded_model.predict_proba(sample_data_scaled)

        # Assign predictions to the original data
        output_data = data.assign(Prediction=predictions)
        output_data['Prediction'] = predictions
        output_data['Probability'] = probabilities.max(axis=1)
        output_data['Prediction'] = output_data['Prediction'].map({0: 'No Stroke', 1: 'Stroke'})

        ###xxx

        # Define a function to apply custom styling to the whole row based on the 'Prediction' column
        def color_prediction_row(row):
            if row['Prediction'] == 'Stroke':
                return ['background-color: tomato'] * len(row)  # Apply color to all columns
            else:
                return ['background-color: lightgreen'] * len(row)  # Apply color to all columns

        # Apply the color styling across the whole row based on 'Prediction' column
        styled_output_data = output_data.style.apply(color_prediction_row, axis=1)

        # Display the styled table in Streamlit
        st.dataframe(styled_output_data)

        ##st.write(output_data)

        # Downloadable CSV
        csv = convert_df(output_data)

        st.download_button(
            label='Download Prediction Results',
            data=csv,
            file_name='Stroke_Prediction_Results.csv',
            mime='text/csv'
        )

        # Visualize results
        prediction_counts = output_data['Prediction'].value_counts()
        colors = ['lightgreen', 'tomato']
        fig = go.Figure(data=[go.Pie(labels=prediction_counts.index,
                                     values=prediction_counts.values, 
                                     hole=.3, textinfo='label+percent')])
        fig.update_traces(marker=dict(colors=colors))

        st.plotly_chart(fig, theme='streamlit')


# -

csv_file = st.file_uploader("Choose a CSV file", type='csv')

if csv_file is not None:
    data = pd.read_csv(csv_file)

if st.button('Batch predict'):
    predict(data)
