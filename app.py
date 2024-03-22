import os
import openai
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")  
 
llm_model="gpt-3.5-turbo-0125"


def get_completion(prompt,model=llm_model):
    messages=[{"role":"user","content":prompt}]
    response=openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content



customer_review="""
Your product is trrrible! I don't know how
you were able to get this to the market.
I don't want this! Actually no one should want this.
Seriously! Give me money now!
"""

tone="""Proper English in a nice, warm, respectful tone"""
language="Portuguese"

promp=f"""
Rewrite the following {customer_review} in a {tone}, and then
please translate the new review message into {language}.
"""

# rewrite=get_completion(prompt=promp)
# print(rewrite)


#using Langchain & prompt templates

chat_model=OpenAI(openai_api_key=OPENAI_API_KEY)


template_string=f"""
Translate the following text {customer_review}
into italiano in a polite tone
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", f"{template_string}"),
    ("user", "{input}")
])

chain = prompt_template | chat_model

chain.invoke({"input": f"{template_string}"})