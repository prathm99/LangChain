import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversational RAG with PDF uploads and Chat History")
st.write("Upload PDFs and chat with their content!")

# input the groq api key
api_key = st.text_input("Enter your Groq API Key:", type="password")

## check if groq api key is provided
if api_key:
    llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant")
    #chat interface
    session_id = st.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}
    
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    # Process uploaded files
    if uploaded_files:
        documents = [] 
        for uploaded_file in uploaded_files:
            tempdf = f"./temp.pdf"
            with open(tempdf, "wb") as f:
                f.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            loader = PyPDFLoader(tempdf)
            docs = loader.load()
            documents.extend(docs)
    
        #split and create embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, collection_name=session_id)
        retriever = vectorstore.as_retriever()

        #prompt
        contextualize_q_system_prompt = (
        "Given a chat history and latest user questions which might refer to previous messages in chat histroy,"
        "formulate a standalone question that can be answered without the chat history."
        "Do NOT answer the question just reformulate it if needed else return as is"
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
        
        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        #Answer question promt
        system_promt = (
            "You are an assistant for Question Answering over a set of documents."
            "Use the following pieces of context to answer the question at the end."
            "If you don't know the answer, just say that you don't know, don't try to make up an answer."
            "Keep the answer concise and to the point."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_promt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
        
        question_answering_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answering_chain)

        def get_sesion_history(session:str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_sesion_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_histroy = get_sesion_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                  "configurable":{"session_id": session_id}
                }
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response["answer"])
            st.write("Chat History:", session_histroy.messages)

else:
    st.warning("Please enter your Groq API Key to proceed.")