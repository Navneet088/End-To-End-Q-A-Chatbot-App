import streamlit as st
import openai 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
#groq_api_key = os.getenv("GROQ_API_KEY")
llm=ChatGroq(model="llama-3.1-8b-instant")


#langchain tracking settings
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACKING_V2']="true"
os.environ['LANGCHAIN_PROJECT']="Simple Q&A Chatbot With Groq LLM"


#prompt template
prompt=ChatPromptTemplate.from_messages(
    [
    ("system","You are a helpful assistant that answer questions based on the context provided."),  
    ("human"," Question: {question}")

    ]
)
def generate_response(question,api_key,llm,temperature,max_tokens):
    openai.api_key=api_key
    llm=ChatGroq(model=llm)
    output_parser=StrOutputParser()
    chain=prompt | llm | output_parser
    answer=chain.invoke({"question":question})
    return answer

st.title("Simple Q&A Chatbot With Groq LLM")
## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API Key:", type="password")
# drop dwon to select model
llm=st.sidebar.selectbox("Select Groq LLM Model:", options=["llama-3.1-8b-instant","llama-3.1-16b-instant"])

#adjust respounce parameters

temperature=st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens=st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

#main interface 
st.write("Ask any question and get an answer from the Groq LLM!")
user_input=st.text_input("you:")

if user_input:
        response=generate_response(user_input,api_key,llm,temperature,max_tokens)
        st.write(f"Assistant: {response}")
else:
      st.write("Please enter a question to get started!")
