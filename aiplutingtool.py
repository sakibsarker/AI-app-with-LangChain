from langchain_community.tools import AIPluginTool
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import ChatOpenAI
tool = AIPluginTool.from_plugin_url("https://www.klarna.com/.well-known/ai-plugin.json")
llm = ChatOpenAI(temperature=0)
tools = load_tools(["requests_all"])
tools += [tool]

agent_chain = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

print(agent_chain.run("what t shirts are available in klarna?"))