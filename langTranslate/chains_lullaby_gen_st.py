import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import streamlit as st

llm = "gpt-3.5-turbo-0125"
openai = ChatOpenAI(temperature=0.9, model=llm)


def generate_lullaby(location, name, language):
    template = """
    As a children's book writer, please come up with a simple and short (90 words)
    lullaby based on the location {location} and the main character {name}
    
    STORY:
    """
    prompt = PromptTemplate(input_variables=["location", "name"], template=template)

    chain_story = LLMChain(llm=openai, prompt=prompt, output_key="story", verbose=True)

    # chain to translate
    template_update = """
    Translate the {story} into {language}. Make sure
    the language is simple and fun.
    
    TRANSLATION:
    """

    prompt_translate = PromptTemplate(
        input_variables=["story", "language"], template=template_update
    )

    chain_translate = LLMChain(
        llm=openai, prompt=prompt_translate, output_key="translated"
    )

    # create teh sequential chain 1-infinity whare have all the chain
    overall_chain = SequentialChain(
        chains=[chain_story, chain_translate],
        input_variables=["location", "name", "language"],
        output_variables=["story", "translated"],  # return story and the translated
        verbose=True,
    )

    response = overall_chain.invoke(
        ({"location": location, "name": name, "language": language})
    )

    return response


def main():
    st.set_page_config(page_title="Generate Childrens Lullaby", layout="centered")
    st.title("Let AI Write and Translate a Lullaby for you")
    st.header("Get Started...")

    location_input = st.text_input(label="Where is the story set?")
    main_character_input = st.text_input(label="What's the  main charater's name")
    language_input = st.text_input(label="Translate the story into...")

    submit_button = st.button("submit")

    if location_input and main_character_input and language_input:
        if submit_button:
            with st.spinner("Generating lullaby.."):
                response = generate_lullaby(
                    location=location_input,
                    name=main_character_input,
                    language=language_input,
                )
                with st.expander("English Version"):
                    st.write(response["story"])
                with st.expander(f"{language_input} Version"):
                    st.write(response["translated"])
            st.success("Lullaby Successfully Generated!")

    pass


if __name__ == "__main__":
    main()
