from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

chat = OpenAI()

prompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt
)

while True:
    content = input("Enter: >> ")

    result = chain.invoke({"content": content})

    print(result["text"])
