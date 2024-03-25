import os
import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import List
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

class VacationInfo(BaseModel):
    leave_time: str=Field(description="When they are leaving.It's usually")
    leave_from: str=Field(description="Where are they leaving from.\
                         it's a city, airport or state, or province")
    cities_to_visit: List=Field(description="The cities, towns they will be visisting on\
                         their trip. This needs to be in a list")
    num_people: int=Field(description="this is an integer for a number of people on this trip")

    @validator('num_people')
    def check_num_people(cls,field):
        if field<=0:
            raise ValueError("Badly formed number")
        return field
    
pydantic_parser=PydanticOutputParser(pydantic_object=VacationInfo)
format_instructions=pydantic_parser.get_format_instructions()


email_template_revised="""
From the following email, extract the following information ragarding
this trip.

email: {email}

{format_instructions}
"""

updated_prompt=ChatPromptTemplate.from_template(template=email_template_revised)
meassages=updated_prompt.format_messages(email=email_response,
                                      format_instructions=format_instructions)
format_response=chat_model.invoke(meassages)
# print(type(format_response.content))

vacation=pydantic_parser.parse(format_response.content)
# print(type(vacation))
print(f"Time: {vacation.leave_time}")
print(f"Leave from: {vacation.leave_from}")
print(f"Total prople: {vacation.num_people}")

for iteam in vacation.cities_to_visit:
    print(f"Cities: {iteam}")