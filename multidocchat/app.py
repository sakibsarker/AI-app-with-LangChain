import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain


load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
 
llm_model="gpt-3.5-turbo-0125"
llm=ChatOpenAI(temperature=0.7,model=llm_model)


pf_loader=PyPDFLoader("./multidocchat/docs/RachelGreenCV.pdf")
documents = pf_loader.load()


#set up qa chain
chain = load_qa_chain(llm, verbose=True)
query = 'Who is the CV about?'
response = chain.run(input_documents=documents,
                     question=query)

print(response)
