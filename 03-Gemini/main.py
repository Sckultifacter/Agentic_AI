# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI

# load_dotenv()

# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.5-flash",
#     temperature=0
# )
# response = llm.invoke("What are AI agents?")
# print(response.content)

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    "google_genai:gemini-3.5-flash"
)

response=model.invoke("What is the captial of India?")

print(response.content[0]['text'])