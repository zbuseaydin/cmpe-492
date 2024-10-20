import getpass
import os
from dotenv import load_dotenv

from typing import Annotated, List

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_core.tools import tool
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional

from langchain_experimental.utilities import PythonREPL
from typing_extensions import TypedDict


from typing import List, Optional
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph, START
from langchain_core.messages import HumanMessage, trim_messages


import functools
import operator

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from IPython.display import Image, display

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

load_dotenv()

_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("TAVILY_API_KEY")


tavily_tool = TavilySearchResults(max_results=5)


# Define the state types
class DebateState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    topic: str  # This should remain a single value
    team_assignments: Dict[str, str]
    current_speaker: str
    debate_stage: str
    team_responses: Dict[str, str]
    next: str

# Define the nodes
def host_node(state: DebateState) -> DebateState:
    current_stage = state["debate_stage"]
    
    if current_stage == "opening_statements":
        if state["current_speaker"] == "Team1":
            state["next"] = "Team2Supervisor"
            state["current_speaker"] = "Team2"
        else:
            state["debate_stage"] = "rebuttal"
            state["next"] = "Team1Supervisor"
            state["current_speaker"] = "Team1"
    elif current_stage == "rebuttal":
        if state["current_speaker"] == "Team1":
            state["next"] = "Team2Supervisor"
            state["current_speaker"] = "Team2"
        else:
            state["debate_stage"] = "closing_statements"
            state["next"] = "Team1Supervisor"
            state["current_speaker"] = "Team1"
    elif current_stage == "closing_statements":
        if state["current_speaker"] == "Team1":
            state["next"] = "Team2Supervisor"
            state["current_speaker"] = "Team2"
        else:
            state["debate_stage"] = "evaluation"
            state["next"] = "Jury"
    
    return state

def team_supervisor_node(state: DebateState, team_name: str) -> DebateState:
    llm = ChatOpenAI(temperature=0.7)
    prompt = ChatPromptTemplate.from_template(
        "You are the supervisor for {team_name} in a debate on {topic}. "
        "Current debate stage: {stage}. "
        "Previous messages: {messages} "
        "Provide a response for your team."
    )
    
    response = llm.invoke(prompt.format_messages(
        team_name=team_name,
        topic=state["topic"],
        stage=state["debate_stage"],
        messages=state["messages"][-5:]  # Last 5 messages for context
    ))
    
    new_state = state.copy()
    new_state["messages"] = state["messages"] + [response]
    new_state["team_responses"] = {**state["team_responses"], team_name: response.content}
    new_state["next"] = "Host"
    
    return new_state

def research_agent_node(state: DebateState, team_name: str) -> DebateState:
    search_results = tavily_tool.run(state["topic"])
    
    llm = ChatOpenAI(temperature=0.5)
    prompt = ChatPromptTemplate.from_template(
        "You are a research agent for {team_name} in a debate on {topic}. "
        "Use the following search results to provide relevant information: {results}"
    )
    
    response = llm(prompt.format_messages(
        team_name=team_name,
        topic=state["topic"],
        results=search_results
    ))
    
    state["messages"].append(response)
    return state

def antithesis_agent_node(state: DebateState, team_name: str) -> DebateState:
    llm = ChatOpenAI(temperature=0.7)
    prompt = ChatPromptTemplate.from_template(
        "You are an antithesis agent for {team_name} in a debate on {topic}. "
        "Generate challenging questions or counterarguments based on the following context: {context}"
    )
    
    response = llm(prompt.format_messages(
        team_name=team_name,
        topic=state["topic"],
        context=state["messages"][-3:]  # Last 3 messages for context
    ))
    
    state["messages"].append(response)
    return state

def summarizer_agent_node(state: DebateState, team_name: str) -> DebateState:
    llm = ChatOpenAI(temperature=0.3)
    prompt = ChatPromptTemplate.from_template(
        "You are a summarizer agent for {team_name} in a debate on {topic}. "
        "Provide a concise summary of the key points from the following context: {context}"
    )
    
    response = llm(prompt.format_messages(
        team_name=team_name,
        topic=state["topic"],
        context=state["messages"][-5:]  # Last 5 messages for context
    ))
    
    state["messages"].append(response)
    return state

def jury_node(state: DebateState) -> DebateState:
    llm = ChatOpenAI(temperature=0.2)
    prompt = ChatPromptTemplate.from_template(
        "You are the jury evaluating a debate on {topic}. "
        "Review the debate and provide a verdict on which team presented a stronger argument. "
        "Debate summary: {summary}"
    )
    
    response = llm(prompt.format_messages(
        topic=state["topic"],
        summary="\n".join([msg.content for msg in state["messages"][-10:]])  # Last 10 messages
    ))
    
    state["messages"].append(response)
    state["next"] = "FINISH"
    return state

# Create the graph
debate_graph = StateGraph(DebateState)

# Add nodes
debate_graph.add_node("Host", host_node)
debate_graph.add_node("Team1Supervisor", lambda s: team_supervisor_node(s, "Team1"))
debate_graph.add_node("Team2Supervisor", lambda s: team_supervisor_node(s, "Team2"))

# Add team-specific agent nodes
for team in ["Team1", "Team2"]:
    debate_graph.add_node(f"{team}ResearchAgent", lambda s: research_agent_node(s, team))
    debate_graph.add_node(f"{team}AntithesisAgent", lambda s: antithesis_agent_node(s, team))
    debate_graph.add_node(f"{team}SummarizerAgent", lambda s: summarizer_agent_node(s, team))

debate_graph.add_node("Jury", jury_node)

# Add edges
debate_graph.add_edge(START, "Host")
debate_graph.add_edge("Host", "Team1Supervisor")
debate_graph.add_edge("Host", "Team2Supervisor")

# Team preparation edges
for team in ["Team1", "Team2"]:
    supervisor = f"{team}Supervisor"
    debate_graph.add_edge(supervisor, f"{team}ResearchAgent")
    debate_graph.add_edge(supervisor, f"{team}AntithesisAgent")
    debate_graph.add_edge(supervisor, f"{team}SummarizerAgent")
    debate_graph.add_edge(f"{team}ResearchAgent", supervisor)
    debate_graph.add_edge(f"{team}AntithesisAgent", supervisor)
    debate_graph.add_edge(f"{team}SummarizerAgent", supervisor)

# Debate flow edges
debate_graph.add_edge("Team1Supervisor", "Host")
debate_graph.add_edge("Team2Supervisor", "Host")

# Final evaluation
debate_graph.add_edge("Host", "Jury")
debate_graph.add_edge("Jury", END)

# Add conditional edges for the Host to manage debate stages
debate_graph.add_conditional_edges(
    "Host",
    lambda x: x["next"],
    {
        "Team1Supervisor": "Team1Supervisor",
        "Team2Supervisor": "Team2Supervisor",
        "Jury": "Jury",
        "FINISH": END,
    }
)

# Compile the graph
debate_chain = debate_graph.compile()
from IPython.display import Image, display

display(Image(debate_chain.get_graph().draw_mermaid_png()))

# Initialize the debate
initial_state = DebateState(
    messages=[HumanMessage(content="Let's start the debate on the topic: AI's impact on job markets")],
    topic="AI's impact on job markets",
    team_assignments={"Team1": "Pro", "Team2": "Con"},
    current_speaker="Team1",
    debate_stage="opening_statements",
    team_responses={},
    next="Host"
)

# Run the debate
final_state = debate_chain.invoke(initial_state)

# Print the debate results
print("Debate Results:")
for message in final_state["messages"]:
    print(f"{message.type}: {message.content}\n")
