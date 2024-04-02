from langchain_openai import ChatOpenAI,OpenAIEmbeddings
import numpy as np

 
llm="gpt-3.5-turbo-0125"
chat = ChatOpenAI(temperature=0.9, model=llm)
embeddings=OpenAIEmbeddings()

text1="Math is a great subject to study"
text2="Dogs are friendly when they are happy and well fed"
text3="Physics is not one of my favorites subjects"


embed1=embeddings.embed_query(text1)
embed2=embeddings.embed_query(text2)
embed3=embeddings.embed_query(text3)

# print(f"Embed1== {embed1}")

similarity=np.dot(embed3,embed2,embed1)
print(f"similarity %: {similarity*100}")