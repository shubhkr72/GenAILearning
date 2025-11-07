from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
# it will download a model of around 700 MB
llm = HuggingFacePipeline.from_model_id(
    model_id='ibm-granite/granite-4.0-h-350m',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.2,
        # max_new_tokens=500
    )
)
model = ChatHuggingFace(llm=llm)


messages=[
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about LangChain')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)
