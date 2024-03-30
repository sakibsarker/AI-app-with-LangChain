import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
 
llm="gpt-3.5-turbo-0125"
chat = ChatOpenAI(temperature=0.9, model=llm)

with open("./data/dream.txt",encoding="utf-8") as papper:
    speech=papper.read()

text_splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len
)

texts=text_splitter.create_documents([speech])
print(texts[0])