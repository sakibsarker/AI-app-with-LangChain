import os
import openai
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")  
 
llm_model="gpt-3.5-turbo-0125"


customer_review="""
Your product is trrrible! I don't know how
you were able to get this to the market.
I don't want this! Actually no one should want this.
Seriously! Give me money now!
"""



#using Langchain & prompt templates

chat_model=OpenAI(temperature=0.7,model=llm_model)


template_string=f"""
Translate the following text {customer_review}
into italiano in a polite tone
"""

prompt_template = ChatPromptTemplate.from_template(template_string)
translation_message = prompt_template.format_prompt(
    customer_review=customer_review
)
response=chat_model(translation_message)

response = chat_model(response)
print(response)