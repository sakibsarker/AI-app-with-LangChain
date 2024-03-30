import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
 
llm="gpt-3.5-turbo-0125"
chat = ChatOpenAI(temperature=0.9, model=llm)

with open("./data/dream.txt",encoding="utf-8") as papper:
    speech=papper.read()

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

docs=text_splitter.create_documents([speech])
print(len(docs))
print(docs[0])
print(docs[1])