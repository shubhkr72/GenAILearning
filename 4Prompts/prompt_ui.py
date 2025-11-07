# from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from dotenv import load_dotenv
load_dotenv()

# it will download a model of around 2GB
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=1,
        max_new_tokens=500
    )
)
model = ChatHuggingFace(llm=llm)


import streamlit as st

st.header("Chat Model Alaska")
user_input=st.text_input("Enter your prompt")

if st.button("Chat"):
    result=model.invoke(user_input).content
    print(result)
    assistant_text = result.split("<|assistant|>")[-1].strip()
    print(assistant_text)

    st.text(assistant_text)
    