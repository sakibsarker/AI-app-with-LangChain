import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")  

llm_model = "gpt-3.5-turbo"
# llm_model="gpt-3.5-turbo-0125"

prompt="How old is the University"

messages = [HumanMessage(content=prompt)]

llm = OpenAI(temperature=0.7)
chat_model=OpenAI(temperature=0.7)


print(llm.invoke("What is the weather in WA DC"))
print("=======")
print(chat_model.invoke(messages))
# print(chat_model.invoke("What is the weather in WA DC"))