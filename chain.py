import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv



load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
 
llm="gpt-3.5-turbo-0125"
chat = ChatOpenAI(temperature=0.9, model=llm)


#llmChain
prompt=PromptTemplate(
    input_variables=["language"],
    template="How do you say good morning in {language}"
)

chain=LLMChain(llm=chat,prompt=prompt)
# print(chain.run(language="France"))
print(chain.invoke({"language": "France"}))
