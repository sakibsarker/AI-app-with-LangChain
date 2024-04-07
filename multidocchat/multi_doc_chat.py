import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import (LLMChain, ConversationalRetrievalChain)
from load_docs import load_docs
import streamlit as st
from streamlit_chat import message

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
 
llm_model="gpt-3.5-turbo-0125"
llm=ChatOpenAI(temperature=0.7,model=llm_model)


#load file
documents =load_docs()
chat_history=[]


#now split the data into chunks
text_splitter=CharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=10
)

docs=text_splitter.split_documents(documents)

#create out vector db

vectordb=Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    persist_directory='/data'
)

vectordb.persist()

qa_chain=ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={'k':6}),
    return_source_documents=True,
    verbose=False
)

#frontend
st.title("DOCS QA Bot Using LangChain")
st.header("Ask anyting about your documents...")

if 'generated' not in st.session_state:
    st.session_state['generated']=[]
if 'past' not in st.session_state:
    st.session_state['past']=[]

def get_query():
    input_text=st.chat_input("Ask a question about your document...")
    return input_text

#retriev the user input

user_input=get_query()
if user_input:
    result=qa_chain({'question':user_input,'chat_history':chat_history})
    st.session_state.past.append(user_input)
    st.session_state.generated.append(result['answer'])

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['generated'][i],key=str(i))
        message(st.session_state['past'][i],is_user=True,key=str(i)+'_user')