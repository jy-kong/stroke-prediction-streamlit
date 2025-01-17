import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(layout="wide", page_title='Stroke Prediction Oracle', page_icon='🔮')


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# +
#local_css("style/style.css")
# -

st.title("Exploratory Data Analysis")

st.markdown("""
    You can upload stroke medical data of your stroke patients here and explore it with our tools!
""")

data = None

def explore(data):
    
    if data is None:
        st.error("Please submit a CSV file before exploration")
    elif data is not None:
        empty_columns = []
        for column in data.columns:
            if data[column].isnull().values.any():
                empty_columns.append(column)
        if len(empty_columns) != 0:
            st.error(f"Your stroke medical data have empty column(s) {empty_columns}. Please fill in all the columns before exploration.")
        else:
            pr = ProfileReport(data, title="YData Profiling Report", explorative=True)
            st_profile_report(pr)

uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

if st.button("Explore"):
    explore(data)
