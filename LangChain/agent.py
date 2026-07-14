#Agent creation with step by step tracer.
#=======================================================
# Step 0- Load env
#=======================================================
from dotenv import load_dotenv

load_dotenv()

#=======================================================
# Step 1- Model Creation
#=======================================================
from langchain.chat_models import init_chat_model
model = init_chat_model(
    "google_genai:gemini-3.5-flash"
)

#=======================================================
# Step 2- Tool Definition
#=======================================================
from langchain_core.tools import tool
import math

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

print('===Available Tools==')
for t in tools:
    print(f"{t.name}:{t.description}")

print()

#=======================================================
# Step 3- Creating the agent("the loop")
#=======================================================
from langchain.agents import create_agent

agent=create_agent(model=model,tools=tools)

#=======================================================
# Step 4- Run the agent(with trace)
#=======================================================
# Iterate through the agent's message history.
# Each message represents a stage of execution (user input, AI tool call,
# tool result, or final AI response), allowing us to visualize how the
# agent arrived at its final answer

def run_agent(question:str):
    """Run the agent and print a clean,beginner friendlt execution trace"""
    print(f"\nUser:{question}")
    print("-"*60)

    result=agent.invoke({
        "messages":[("user",question)]
    })

    print("Clean Agent execution trace")
    print("-"*60)

    step=1

    for msg in result["messages"]:
        if msg.type=="human":
            print(f"{step}.User asked:")
            print(f"{msg.content}")
            step+=1
        
        elif msg.type=="ai" and getattr(msg,"tool_calls",None):
            for tool_call in msg.tool_calls:
                tool_name=tool_call["name"]
                tool_args=tool_call["args"]
                print(f"Agent Decision:")
                print(f"I need to use the tool:{tool_name}")
                print(f"Tool input:{tool_args}")
                step+=1

        elif msg.type=="tool":
            print(f"{step}.Tool Observation")
            print(f"Tool Returned:{msg.content}")
            step+=1
        
        elif msg.type=="ai" and msg.content:
            print(f"{step}. Final answer:")
            print(f"{msg.content[0]['text']}")
            step+=1


#=======================================================
# Test Cases
#=======================================================
#run_agent("What is 42+58?")
run_agent("What is 12 multiplied by 30 and divided by 3?")