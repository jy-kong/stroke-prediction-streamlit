import streamlit as st
import requests
from streamlit_lottie import st_lottie

# --------------------------------------------About & Credits -----------------------------------------------------

st.subheader('About :question:')

def load_lottieurl(url: str):
    import requests  # Ensure requests module is imported
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_about = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_sy6jjyct.json")

col1, col2 = st.columns([6, 3])
with col1:
    st.write("""
        The purpose of this platform is to assist healthcare professionals and researchers in analyzing data for early stroke prediction.
        \nThis website offers several functionalities:
        \n**Stroke Prediction**: Train and deploy machine learning models to predict stroke occurrence.
        \n**Feature Analysis**: Explore the influence of demographic, clinical, and lifestyle factors on stroke risk.
        \n**Comparison of Models**: Evaluate and compare the performance of various machine learning models.
        \n**EDA**: Perform Exploratory Data Analysis on the provided dataset.
        \n**About & Credits**: Share your feedback and suggestions for future improvements!
    """)
with col2:
    st_lottie(
        lottie_about,
        height=340,
        width=None,
        quality="high",
        key=None,
    )

"---"

st.subheader('Documentation :memo:')
st.write("""The Website User Manual is available here.""")
st.write("""[User Manual](https://drive.google.com/drive/folders/1JBCusrKUT6xpfowFHWOB2Gjs6pTr2lm2?usp=sharing)""")

"---"

st.subheader('Credits :star2: :computer:')
st.write("""\nThis website was developed by [Your Name] from [Your Institution].
            \nThe project was supervised by [Your Supervisor's Name and Title].""")

"---"

st.subheader(":mailbox: Get In Touch With Me!")
st.write("""\nIf you have any feedback, don't hesitate to fill in this form.""")

contact_form = """
<form action="https://formsubmit.co/your-email@example.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Your name" required>
     <input type="email" name="email" placeholder="Your email" required>
     <textarea name="message" placeholder="Your feedback here"></textarea>
     <button type="submit">Send</button>
</form>
"""
st.markdown(contact_form, unsafe_allow_html=True)
