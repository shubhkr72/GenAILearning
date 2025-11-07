from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# it will download a model of around 2GB
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.2,
        # max_new_tokens=500
    )
)
model = ChatHuggingFace(llm=llm)

chat_history=[]

print("Chat with Chatgpt want to exit type exit")
while True:
    user_input=input("You: ")
    chat_history.append(user_input)
    if user_input=='exit':
        
        break
    result=model.invoke(chat_history).content
    assistant_text = result.split("<|assistant|>")[-1].strip()
    chat_history.append(assistant_text)
    print("AI: ",assistant_text)


with open("chat.text", "w", encoding="utf-8") as f:
    f.write(chat_history)