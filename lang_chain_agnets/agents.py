# agents.py
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph
from typing import Dict
import os

# Shared state type (plain dict works fine)
class MultiAgentState(dict):
    query: str
    research_summary: str
    draft_article: str
    reviewed_article: str

# Research Agent
def research_agent(state: MultiAgentState):
    print("🔍 Research agent running...")
    search = DuckDuckGoSearchRun()
    results = search.run(state["query"])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"Summarize key insights from the following search results:\n\n{results}"
    response = llm.invoke(prompt)
    # .content holds the model text depending on LLM wrapper
    state["research_summary"] = getattr(response, "content", str(response))
    return state

# Writer Agent
def writer_agent(state: MultiAgentState):
    print("✍ Writer agent running...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
    prompt = (
        f"Write a short beginner-friendly article using the following research:\n\n"
        f"{state['research_summary']}\n\n"
        "Use headings and simple language."
    )
    response = llm.invoke(prompt)
    state["draft_article"] = getattr(response, "content", str(response))
    return state

# Critic Agent
def critic_agent(state: MultiAgentState):
    print("🧑‍⚖ Critic agent running...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = (
        f"Review the article below for clarity, accuracy, and flow. Provide "
        f"concise feedback and an improved version formatted with 'Feedback:' and 'Improved Article:'.\n\n"
        f"{state['draft_article']}"
    )
    response = llm.invoke(prompt)
    state["reviewed_article"] = getattr(response, "content", str(response))
    return state

# Build graph once and compile (reusable)
def build_multi_agent_graph():
    graph = StateGraph(MultiAgentState)
    graph.add_node("research", research_agent)
    graph.add_node("writer", writer_agent)
    graph.add_node("critic", critic_agent)
    # edges
    graph.add_edge("research", "writer")
    graph.add_edge("writer", "critic")
    graph.set_entry_point("research")
    graph.set_finish_point("critic")
    compiled = graph.compile()
    return compiled