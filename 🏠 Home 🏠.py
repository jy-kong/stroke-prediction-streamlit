import streamlit as st
import pandas as pd

st.set_page_config(
    page_title='Stroke Prediction Oracle',
    layout='wide',
    page_icon="🔮"
)


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# +
#local_css("style/style.css")
# -

st.title("Stroke Prediction Oracle :crystal_ball:")

st.markdown(
    """
    #### Introduction
    Welcome to the main page of Stroke Prediction Oracle. This web application is a prototype designed to 
    assist **medical professionals**, **stroke patients**, and **researchers** in predicting and analyzing 
    stroke risks. The **Random Forest** model serves as the predictive engine in this application. 
    Stroke Prediction Oracle offers three key functionalities: **Exploratory Data Analysis**, 
    **Single Prediction**, and **Multiple/Dataset Prediction**. 

    Users can upload datasets containing patient stroke risk data and explore insights on the 
    **Exploratory Data Analysis** page. Additionally, users can manually input medical data of an individual 
    on the **Single Prediction** page or upload a CSV file on the **Multiple/Dataset Prediction** page 
    to evaluate the stroke risk of multiple patients efficiently.

    #### What is Stroke?
    Stroke is a serious medical condition that occurs when blood flow to the brain is interrupted, 
    either due to a blocked artery (ischemic stroke) or a ruptured blood vessel (hemorrhagic stroke). 
    This disruption deprives brain cells of oxygen and nutrients, leading to cell death within minutes. 
    Stroke can result in severe physical and cognitive impairments, including paralysis, speech difficulties, 
    memory loss, and emotional challenges. Risk factors include high blood pressure, diabetes, smoking, 
    obesity, and a sedentary lifestyle.

    #### Video About Stroke Facts
    """
)
col1, col2 = st.columns(2)

video_file = open("What Causes A Brain Stroke? | Brain Attack | The Dr Binocs Show | Peekaboo Kidz.mp4", 'rb')
video_bytes = video_file.read()

with col1: 
    st.video(video_bytes)


st.markdown("""
    
    #### What are the stroke prediction outcomes that will be assigned to the patients when using this predictor?
    The prediction outcomes that will be assigned to patients and their respective definitions are listed below:

    | Prediction Outcome | Definition |
    | ------------------- | ----------- |
    | Stroke              | Indicates that the patient has a high risk of experiencing a stroke based on the input medical and lifestyle data. |
    | No Stroke           | Indicates that the patient has a low risk of experiencing a stroke based on the input medical and lifestyle data. |

    
    #### User Manual
    To understand how to use the Stroke Prediction Oracle in detail, please read the [Stroke Prediction Oracle User Manual](https://drive.google.com/file/d/13Q-lUBe5uA9yyu1syBOolFjgnlo-OoKq/view?usp=sharing).
    
"""
)

