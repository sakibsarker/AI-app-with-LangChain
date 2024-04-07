import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline
import requests
import streamlit as st
from IPython.display import Audio
# Audio(audio, rate=sampling_rate)

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY=os.getenv("HUGGINGFACE_API_KEY")

llm_model="gpt-3.5-turbo-0125"
llm=ChatOpenAI(temperature=0.7,model=llm_model)

#1image to text implementation
def image_to_text(url):
    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large",max_new_tokens=1000)
    text=pipe(url)[0]['generated_text']
    print(f"Image Captioning::{text}")
    return text


#2llm generate recipe from image text
def text_to_speech(text):
    API_URL = "https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload ={
	"inputs": text,
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content



#3 text to voice

def main():
    image_file_path = os.path.join(os.path.dirname(__file__), "ingredients.jpeg")
    caption = image_to_text(image_file_path)
    audio=text_to_speech(text=caption)
    # print(caption)
    with open("audio.flac","wb") as file:
        file.write(audio)
    print("Audio file successfully generated.")

if __name__=='__main__':
    main()   