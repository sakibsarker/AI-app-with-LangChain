from langchain_openai import OpenAI
from langchain.agents import Tool,initialize_agent,load_tools
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm_model = "gpt-3.5-turbo-0125"
llm = OpenAI(temperature=0.0)
 
prompt=PromptTemplate(
    input_variables=["query"],
    template="{query}"
)

llm_chain=LLMChain(llm=llm,prompt=prompt)

llm_tool=Tool(
    name="Language Model",
    func=llm_chain.run,
    description="Use this tool for general queries and logic"
)

tools = load_tools(
    ['llm-math'],
    llm=llm
)

tools.append(llm_tool) #adding the new tool to our tools list

print(tools[0].name, tools[0].description)


agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3
)

query = "If i have 54 eggs and Mary has 10, and 5 more people have 12 eggs each. \
    How many eggs to we have in total?"

result = agent.invoke(query)

print(result['output'])

