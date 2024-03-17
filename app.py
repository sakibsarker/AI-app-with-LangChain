import os
from dotenv import load_dotenv
from langchain_openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")  

# llm = OpenAI(openai_api_key=OPENAI_API_KEY)

llm_model = "gpt-3.5-turbo"

llm = OpenAI(temperature=0.7)

print(llm.invoke("What is the weather in WA DC"))