import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")

# prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI assistant that helps people find information, you can access all the web sources to find the information."),
    ('user', 'Question:{question}')
    ])

# streamlit app
st.title("Ollama(Gemma:2b) Geni AI App")

input_text = st.text_input("Hey What's going on your mind?")


# ollama model
llm = Ollama(model="gemma:2b")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
