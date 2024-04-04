import os
from langchain_openai import OpenAI
from langchain.agents import Tool,initialize_agent,load_tools
from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv



load_dotenv()
SERP_API_KEY = os.getenv("SERPAPI_API_KEY")


llm_model = "gpt-3.5-turbo-0125"
llm = OpenAI(temperature=0.0)


search=SerpAPIWrapper(serpapi_api_key=SERP_API_KEY)

tools=[
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="google search"
    )
]

self_ask_with_search=initialize_agent(
    tools,
    llm,
    agent='self-ask-with-search',
    verbose=True
)

query = "who has travelled the most: Justin Timberlake, Alicia Keys, or Jason Mraz?"
result=self_ask_with_search.invoke(query)
print(result)