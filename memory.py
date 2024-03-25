import os
import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")  
 
llm_model="gpt-3.5-turbo-0125"
llm=ChatOpenAI(temperature=0.7,model=llm_model)


memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

conversation.predict(input="My name is Sakib. What is yours?")
conversation.predict(input="Great! What's my name?")
conversation.predict(input="Thank you")
