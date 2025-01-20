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
        The purpose of this platform is to assist healthcare professionals and researchers in analyzing data for stroke prediction.
        \nThis website offers several functionalities:
        \n**🔭 Exploratory Data Analysis**: Gain insights from the dataset through detailed EDA.
        \n**🩺 Single Prediction**: Explore the influence of demographic, clinical, and lifestyle factors on stroke risk.
        \n**🔬 Multiple | Dataset Prediction**: Perform batch prediction from a dataset file, displaying outcomes for all records.
        \n**📌 About | Credits**: Share your feedback and suggestions for future improvements!
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
st.write("""[User Manual](https://drive.google.com/file/d/1ZgRSNgZuRfy5swKIYNcgZ0A8-TgWFOGy/view?usp=share_link)""")

"---"

st.subheader('Credits :star2: :computer:')
st.write("""\nThis website was developed by Kong Jing Yuaan from the Faculty of Computer Science & Information Technology at Universiti Malaya.
            \nThe project was supervised by Ts. Dr. Mohd Shahrul Nizam Bin Mohd Danuri.""")

"---"

st.subheader(":mailbox: Get In Touch With Me!")
st.write("""\nIf you have any feedback, don't hesitate to fill in this form.""")

contact_form = """
<form action="https://formsubmit.co/kjyuaan8@gmail.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Your name" required>
     <input type="email" name="email" placeholder="Your email" required>
     <textarea name="message" placeholder="Your feedback here"></textarea>
     <button type="submit">Send</button>
</form>
"""
st.markdown(contact_form, unsafe_allow_html=True)

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# +
local_css("style/style.css")
