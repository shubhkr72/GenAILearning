from langchain_anthropic import ChatAnthropic

from dotenv import load_dotenv
load_dotenv()

model=ChatAnthropic(model='claude-3-5-sonnet',temperature=1.5,max_completion_tokens=100)
query=""
result=model.invoke(query)
print(result)
print(result.content)
