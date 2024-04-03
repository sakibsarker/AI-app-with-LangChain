from langchain_openai import OpenAI
from langchain.agents import Tool,initialize_agent,load_tools
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

llm_model = "gpt-3.5-turbo-0125"
llm = OpenAI(temperature=0.0)

#memory
memory = ConversationBufferMemory(memory_key="chat_history")
#second generic tool
 
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

#conversational agent
conversational_agent=initialize_agent(
    agent="conversational-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_interactions=3,
    memory=memory
)

query = "How old is a person born in 1917 in 2023"

query_two="How old would that person be if their age is multiplied by 100?"
print(conversational_agent.agent.llm_chain.prompt.template)
result = conversational_agent.invoke(query)
results = conversational_agent.invoke(query_two)
# print(result['output'])
