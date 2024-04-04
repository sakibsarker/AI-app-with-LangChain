from langchain_openai import OpenAI
from langchain.agents import Tool,initialize_agent,load_tools
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.docstore import Wikipedia
from langchain.agents.react.base import DocstoreExplorer

llm_model = "gpt-3.5-turbo-0125"
llm = OpenAI(temperature=0.0)
docstore=DocstoreExplorer(Wikipedia())

tools=[
    Tool(
        name="Search",
        func=docstore.search,
        description="search wikipedia"
    ),
    Tool(
        name="Lookup",
        func=docstore.lookup,
        description="lookup a term in wikipedia"
    )
]


docstore_agent=initialize_agent(
    tools,
    llm,
    agent="react-docstore",
    verbose=True,
    max_interations=4
)

query="What ware Einstein's main beliefs?"
result=docstore_agent.invoke(query)

print(docstore_agent.agent.llm_chain.prompt.template)