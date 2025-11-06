from langchain_openai import OpenAI 
from dotenv import load_dotenv
load_dotenv()

model=OpenAI(model='gpt-3')

result=model.invoke("hii how are you")

print(result)