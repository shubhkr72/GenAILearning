from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-1.5-flash-001')

result=model.invoke("how are you")

print(result.content)
print()