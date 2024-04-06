import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
 
llm_model="gpt-3.5-turbo-0125"
llm=ChatOpenAI(temperature=0.7,model=llm_model)


pf_loader=PyPDFLoader("./multidocchat/docs/RachelGreenCV.pdf")
documents = pf_loader.load()


#now split the data into chunks
text_splitter=CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs=text_splitter.split_documents(documents)

#create out vector db

vectordb=Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    persist_directory='/data'
)

vectordb.persist()

#use retrievalQA chain to get info from the vectorstore

qa_chain=RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(search_kwargs={'k':7}),
    return_source_documents=True
)

result=qa_chain.invoke({"query":"Who is the CV about?"})
# results=qa_chain({'query':'Who is the CV about?'})

print(result['result'])