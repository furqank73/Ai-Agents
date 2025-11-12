from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# 🔹 Load environment variables
load_dotenv()

# 🔹 Set up keys
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Groq-Agent-Demo"  # ✅ Project name for LangSmith

# 🔹 Define state
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 🔹 Initialize model
model = ChatGroq(model="llama-3.1-8b-instant")

# 🔹 Default graph (single-step LLM)
def make_default_graph():
    graph = StateGraph(State)

    def call_model(state):
        return {"messages": [model.invoke(state["messages"])]}

    graph.add_node("agent", call_model)
    graph.add_edge("agent", END)
    graph.add_edge(START, "agent")

    agent = graph.compile()
    return agent

# 🔹 Tool-calling agent
def make_alternative_graph():
    """Make a tool-calling agent"""

    @tool
    def add(a: float, b: float):
        """Adds two numbers."""
        return a + b

    # Define tool node and model binding
    tool_node = ToolNode([add])
    model_with_tools = model.bind_tools([add])

    # Node: call model
    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    # Decide whether to continue or end
    def should_continue(state: State):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END

    # Create graph
    graph = StateGraph(State)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.add_edge("tools", "agent")
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)

    agent = graph.compile()
    return agent


# ✅ Create and run agent
if __name__ == "__main__":
    agent = make_alternative_graph()

    # Example input message
    messages = [HumanMessage(content="Add 25 and 45")]
    result = agent.invoke({"messages": messages})

    print("Final Output:")
    print(result)
