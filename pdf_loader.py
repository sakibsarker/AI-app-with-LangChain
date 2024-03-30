import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate,ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
 
llm="gpt-3.5-turbo-0125"
chat = ChatOpenAI(temperature=0.9, model=llm)


loader=PyPDFLoader("./data/react-paper.pdf")
pages = loader.load_and_split()
print(len(pages))
page=pages[0]
# print(page)
print(page.page_content[0:700])
print(page.metadata)