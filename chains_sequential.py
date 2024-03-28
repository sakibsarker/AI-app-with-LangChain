import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,SequentialChain
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


chain_story=LLMChain(
    llm=openai,
    prompt=prompt,
    output_key="story",
    verbose=True
)

story=chain_story.invoke({"location": "Zanzibar","name":"Maya"})

#chain to translate
template_update="""
Translate the {story} into {language}. Make sure
the language is simple and fun.

TRANSLATION:
"""

prompt_translate=PromptTemplate(
    input_variables=["story","language"],
    template=template_update
)

chain_translate=LLMChain(
    llm=openai,
    prompt=prompt_translate,
    output_key="translated"
)

#create teh sequential chain 1-infinity whare have all the chain
overall_chain=SequentialChain(
    chains=[chain_story,chain_translate],
    input_variables=["location","name","language"],
    output_variables=["story","translated"], #return story and the translated
    verbose=True
)

response=overall_chain.invoke(({"location":"Magical Land",
                         "name":"Karyna",
                         "language":"Italian"
                         }))

print(f"English Version ====> \n{response['story']} \n \n")
print(f"Translated Version ====> \n{response['translated']} \n \n")