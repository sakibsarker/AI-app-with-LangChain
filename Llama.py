import os
from dotenv import load_dotenv
from llamaapi import LlamaAPI
from langchain_experimental.llms import ChatLlamaAPI
from langchain.chains import create_tagging_chain

load_dotenv()
Llama_API_KEY = os.getenv("Llama_API_KEY")

llama = LlamaAPI(Llama_API_KEY)


model = ChatLlamaAPI(client=llama)

schema = {
    "properties": {
        "sentiment": {
            "type": "string",
            "description": "the sentiment encountered in the passage",
        },
        "aggressiveness": {
            "type": "integer",
            "description": "a 0-10 score of how aggressive the passage is",
        },
        "language": {"type": "string", "description": "the language of the passage"},
    }
}

chain = create_tagging_chain(schema, model)

print(chain.invoke("give me your money"))