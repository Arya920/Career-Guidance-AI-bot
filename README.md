# Career Buddy - Career Guidance AI Chatbot

## Overview
This project presents the development of a career counseling bot that utilizes Quora data to 
provide users with valuable insights and recommendations for their career choices

## Objective
The main aim of this project is to demonstrate the effectiveness of leveraging Quora data for career 
counseling purposes. The Career Counseling Bot provides accurate and up-to-date guidance, 
assisting users in navigating the complexities of the job market. By tapping into the diverse 
perspectives and insights of the Quora community, the chatbot offers a valuable resource for 
individuals seeking career advice.

## Chatbot Model
The chatbot uses natural language processing and a neural network to understand user queries and provide appropriate responses. The query is broken down into individual words and matched against stored patterns and knowledge. The model also uses a convolutional neural network to capture complex patterns and relationships within the data. The output scores are transformed into probabilities using activation functions like softmax. The model uses stochastic gradient descent during the training process to minimize loss and improve accuracy. By integrating the CNN model, the chatbot provides users with more precise and contextually relevant guidance for their career queries.

## Technologies & Tools Used 
- *Webscrapping using selenium*
- *Python*
- *Machine Learning*
- *NLP*
- *Streamlit*

## Setup Instructions

### Prerequisites
- Python 3.10 (required)
- Git

### Step 1: Clone the Repository
bash
git clone https://github.com/Arya920/Career-Guidance-AI-bot.git
cd Career-Guidance-AI-bot

### Step 2: Set Up Virtual Environment
This project requires Python 3.10. Create and activate a virtual environment:

bash
# Create virtual environment
py -3.10 -m venv myenv310

# Activate virtual environment
myenv310\Scripts\activate
 

### Step 3: Install Dependencies
bash
pip install -r requirements.txt
 

### Step 4: Run the Application
bash
streamlit run Chatbot_frontend.py
 

The application will open in your default web browser