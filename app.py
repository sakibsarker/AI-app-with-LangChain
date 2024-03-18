import os
import openai
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")  

llm_model="gpt-3.5-turbo-0125"
chat_model=OpenAI(temperature=0.7,model=llm_model)


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


promp=f"""
Rewrite the following {customer_review} in a polite tone, and then
please translate the new review message into Portuguese.
"""

rewrite=get_completion(prompt=promp)
print(rewrite)