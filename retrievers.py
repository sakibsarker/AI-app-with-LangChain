import os
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
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


vectorstore.persist()

vector_store=Chroma(persist_directory=persist_directory,
                    embedding_function=embeddings)


retriever =vector_store.as_retriever(search_kwargs={"k":2})
docs=retriever.get_relevant_documents("Tell me more about ReAct prompting")
print(docs[0].page_content)


#make a chain for answer questions
qa_chain=RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=retriever,
    verbose=True,
    return_source_documents=True
)

def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

query="tell me more about ReAct prompting"
llm_response=qa_chain.invoke(query)
print(process_llm_response(llm_response=llm_response))