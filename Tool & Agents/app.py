import streamlit as st
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, ArxivQueryRun, WikipediaQueryRun
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory

import os
from dotenv import load_dotenv


api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchRun(name="Search",description="Use for current events or real-time information")
st.title("🦜🔗 Groq Agent with Streamlit")

"""Simple App to using Streamlit callback handler to display thoughts and actions of an Agent powered by Groq LLM."""

# sidebar settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

if not api_key:
    st.warning("Please enter your Groq API key")
    st.stop()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm a Groq-powered agent. How can I assist you today?"}
    ]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant",streaming=True)

    tools = [search, wiki, arxiv]

    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        handling_parsing_errors=True)
    
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_callback]) # use prompt insider run if you directly want the output
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
