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
    memory=memory
)

conversation.predict(input="My there, I am Sakib")
conversation.predict(input="Why is the sky blue?")
conversation.predict(input="If phenomenon called Rayleigh didn't exist, what color would the sky be?")
conversation.predict(input="What's my name?")

print(memory.load_memory_variables({}))
