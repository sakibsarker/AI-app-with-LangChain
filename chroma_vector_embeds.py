import os
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import numpy as np
 
llm="gpt-3.5-turbo-0125"
chat = ChatOpenAI(temperature=0.9, model=llm)
embeddings=OpenAIEmbeddings()

loader=PyPDFLoader("./data/react-paper.pdf")
docs = loader.load()

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

splits=text_splitter.split_documents(docs)


persist_directory="./data/db/chroma"


vectorstore=Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=persist_directory
)

# print(vectorstore._collection.count())


query="what do they say about ReAct prompting method?"

docs_resp=vectorstore.similarity_search(query=query,k=3)
print(len(docs_resp))
print(docs_resp[0].page_content)

vectorstore.persist()