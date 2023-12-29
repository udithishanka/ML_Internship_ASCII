from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import pickle
import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from htmlTemplates import bot_template, user_template
from langchain.llms import HuggingFaceHub
import time

# Text Extracting from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    #for pdf in pdf_docs:
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Splitting the extracted text into Chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Creating a vectorstore with embeddings
def get_vectorstore(text_chunks,store_name):
    if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                st.session_state.vectorstore1 = pickle.load(f)
                st.write("Pickle Loaded")
                
    else:
            start_time = time.time()
            #embeddings = OpenAIEmbeddings()
            embeddings = OllamaEmbeddings(model="llama2:13b",)
            #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
            st.session_state.vectorstore1 = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(st.session_state.vectorstore1, f)
                st.write("Pickle Computed")
                st.write(time.time()-start_time, "seconds to embed")

    return st.session_state.vectorstore1



def get_conversation_chain(vectorstore):
    
    #llm = ChatOpenAI()
    llm = ChatOllama(model="llama2:13b")
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
    )
    
    return conversation_chain


def handle_userinput(user_question):
    start_time1 = time.time()
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
    st.write(time.time()-start_time1, "seconds to generate response")