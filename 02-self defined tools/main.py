from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent
import math

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0
)

@tool
def add(a:float,b:float)->float:
    """Add two numbers together.Use for addition"""
    return a+b

@tool
def multiply(a:float,b:float)->float:
    """Multiply two numbers together"""
    return a*b

@tool
def divide(a:float,b:float)->float:
    """Divide two numbers.Return error if divided by zero."""
    if(b==0):
        raise ValueError("cannot divide by zero")
    return a/b

@tool
def square_root(a:float)->float:
    """Squared root of a given number.Error if number is negative."""
    if(a<0):
        raise ValueError("cannot find square root of negative numbers")
    else:
        return math.sqrt(a)
    
tools=[add,multiply,divide,square_root]

agent=create_agent(model=model,tools=tools)

def run_agent():
    while True:
        query = input("\nYou: ")

        if query.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break

        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            }
        )
        print("Agent:", response["messages"][-1].content)

if __name__=="__main__":
    run_agent()

    
    