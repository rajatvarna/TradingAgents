from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    a: int
    b: int
    c: int
    count: Annotated[int, operator.add]

def node_a(state):
    return {"a": 1, "count": 1}

def node_b(state):
    return {"b": 1, "count": 1}

def join_node(state):
    return {"c": 1}

workflow = StateGraph(State)
workflow.add_node("A", node_a)
workflow.add_node("B", node_b)
workflow.add_node("Join", join_node)

workflow.add_edge(START, "A")
workflow.add_edge(START, "B")
workflow.add_edge("A", "Join")
workflow.add_edge("B", "Join")
workflow.add_edge("Join", END)

app = workflow.compile()
print(app.invoke({"a": 0, "b": 0, "c": 0, "count": 0}))
