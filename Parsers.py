import os
import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")  
 
llm_model="gpt-3.5-turbo-0125"
chat_model=ChatOpenAI(temperature=0.7,model=llm_model)
# chat_model=ChatOpenAI(openai_api_key=OPENAI_API_KEY)



email_response = """
Here's our itinerary for our upcoming trip to Europe.
We leave from Denver, Colorado airport at 8:45 pm, and arrive in Amsterdam 10 hours later
at Schipol Airport.
We'll grab a ride to our airbnb and maybe stop somewhere for breakfast before 
taking a nap.

Some sightseeing will follow for a couple of hours. 
We will then go shop for gifts 
to bring back to our children and friends.  

The next morning, at 7:45am we'll drive to to Belgium, Brussels - it should only take aroud 3 hours.
While in Brussels we want to explore the city to its fullest - no rock left unturned!

"""

email_template = """
From the following email, extract the following information:

leave_time: when are they leaving for vacation to Europe. If there's an actual
time written, use it, if not write unknown.

leave_from: where are they leaving from, the airport or city name and state if
available.

cities_to_visit: extract the cities they are going to visit. 
If there are more than one, put them in square brackets like '["cityone", "citytwo"].

Format the output as JSON with the following keys:
leave_time
leave_from
cities_to_visit

email: {email}
"""

prompt_template = ChatPromptTemplate.from_template(email_template)
messages = prompt_template.format_messages(email=email_response)
response = chat_model.invoke(messages)

# print(response.content)

#Structured output parser

leave_time_schema=ResponseSchema(name="leave_time", 
                                description="When they are leaving. \
                                It's usually a numberical time of the day. \
                                If no available wirte n/a"
                                )
leave_from_schema=ResponseSchema(name="leave_from", 
                                description="When they are from. \
                                It's a city, airport or state if province."
                                )
cities_to_visit_schema=ResponseSchema(name="cities_to_visit", 
                                description="The cities, towns they will be visit \
                               their trip. This needs to be in a list"
                                )

response_schemas = [
    leave_time_schema,
    leave_from_schema,
    cities_to_visit_schema
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()

email_template_revised = """
From the following email, extract the following information:

leave_time: when are they leaving for vacation to Europe. If there's an actual
time written, use it, if not write unknown.

leave_from: where are they leaving from, the airport or city name and state if
available.

cities_to_visit: extract the cities they are going to visit. If there are more than 
one, put them in square brackets like '["cityone", "citytwo"].

Format the output as JSON with the following keys:
leave_time
leave_from
cities_to_visit

email: {email}
{format_instructions}
"""

updated_prompt=ChatPromptTemplate.from_template(template=email_template_revised)
messages=prompt_template.format_messages(email=email_response,
                                         format_instructions=format_instructions)

response=chat_model.invoke(messages)
print(type(response.content))

output_dict=output_parser.parse(response.content)
print(type(output_dict))
print(f"Cities:::: {output_dict['cities_to_visit']}")