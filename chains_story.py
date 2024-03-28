import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv



load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
 
llm="gpt-3.5-turbo-0125"
openai = ChatOpenAI(temperature=0.9, model=llm)

template="""
As a children's book writer, please come up with a simple and short (90 words)
lullaby based on the location
{location}
and the main character {name}

STORY:
"""

prompt=PromptTemplate(
    input_variables=["location","name"],
    template=template
)

# chain_story=LLMChain(llm=openai,prompt=prompt,verbose=True)
chain_story=LLMChain(llm=openai,prompt=prompt,verbose=True)
story=chain_story.invoke({"location": "Zanzibar","name":"Maya"})
print(story['text'])
