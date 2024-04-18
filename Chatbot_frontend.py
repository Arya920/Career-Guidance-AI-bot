# <========================================================= Importing Required Libraries & Functions =================================================>
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import nltk
nltk.download('stopwords')


nltk.download("punkt")
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import load_model

from PIL import Image
# <-------------------------------------------------------------Functions ----------------------------------------------------------------------------------->
# from chatbot_final_code import clean_up_sentence
# from chatbot_final_code import bow
# from chatbot_final_code import predict_class
# from chatbot_final_code import chatbot_response

# <---------------------------------------------------------- Page Configaration ----------------------------------------------------------------------------->
im = Image.open('bot.jpg')
st.set_page_config(layout="wide",page_title="Student's Career Counselling Chatbot",page_icon = im)




# <---------------------------------------------------------- Main Header ------------------------------------------------------------------------------------->
st.markdown(
    """
    <div style="background-color: #FF8C00 ; padding: 10px">
        <h1 style="color: brown; font-size: 48px; font-weight: bold">
           <center> <span style="color: black; font-size: 64px">C</span>areer <span style="color: black; font-size: 64px">B</span>uddy <span style="color: black; font-size: 64px">
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

# <========================================================= Importing Data Files  ====================================================================>

with open('intents3.json', 'r') as file:
    intents = json.load(file)
with open('words.pkl', 'rb') as file:
    words = pickle.load(file)
with open('classes.pkl', 'rb') as file:
    classes = pickle.load(file)



# <--------------------------hide the right side streamlit menue button --------------------------------->
# Referance ~ "https://towardsdatascience.com/5-ways-to-customise-your-streamlit-ui-e914e458a17c"
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)


# <=========================================================== Sidebar ================================================================================> 
with st.sidebar:
    st.title('''🤗💬 Student's career counselling bot''')
    
    st.markdown('''
    ## About~
    This app has been developed by 5 students of VIT-AP :\n
    Harshita Bajaj [22MSD7013]\n
    Arya Chakraborty [22MSD7020] \n
    Rituparno Das [22MSD2027]\n
    Shritama Sengupta [22MSD7032]\n
    Arundhuti Chakraborty [22MSD7046]

    ''')
    add_vertical_space(5)
    
# <============================================================= Initializing Session State ==========================================================>
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm an AI Career Counselor, How may I help you?"]

if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']



input_container = st.container()

colored_header(label='', description='', color_name='blue-30')
response_container = st.container()


#<================================================== Function for taking user provided prompt as input ================================================>
# def pressed_enter_key(text):
#     if text == 

def get_text():
    input_text = st.text_input("You: ",  key="input", on_change=None)
    return input_text

styl = f"""
<style>
    .stTextInput {{
    position: fixed;
    bottom: 20px;
    z-index: 20;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)

#<------------------------------------------------ Applying the user input box ------------------------------------------------------------------------>
with input_container:
    user_input = get_text()

# <================================================ Loading The Model ===============================================================>
model=load_model('chatbot_model.h5')


# <============================== Function for taking user prompt as input followed by producing AI generated responses ============>


# TODO: Implement a data model that will make a response

# def generate_response(prompt):
#     clean_up_sentence(prompt) # For Lemmatizing and tokenizing the new sentence
#     bow(prompt, words, show_details=True) #
#     predict_class(prompt,model)
#     response = chatbot_response(prompt)
#     return response
#<--------------------Creating the submit button and changing it using CSS----------------------->    
submit_button = st.button("Enter")
styl = f"""
    <style>
        .stButton {{
        position: fixed;
        font-weight: bold;
        margin-top: -10px;
        bottom: 20px;
        left: 1213px;
        font-size: 24px;
        z-index: 9999;
        border-radius: 20px;
        height:200px
        width:100px
        }}
        
    </style>
    """
st.markdown(styl, unsafe_allow_html=True)

#<====================== Conditional display of AI generated responses as a function of user provided prompts=====================================>
with response_container:
    if user_input: 
        if submit_button:
            if user_input == "Who is your maker":
                response = "GOD !!"
                st.session_state.past.append(user_input)


                # TODO: RENDER a response HERE
                st.session_state.generated.append(response)
                #st.text_input("Enter your input", value="", key="user_input")

            else:
                
                response = None # generate_response(user_input)
                st.session_state.past.append(user_input)
                st.session_state.generated.append(response)
                #st.text_input("Enter your input", value="", key="user_input")
        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state['generated'][i], key=str(i))
