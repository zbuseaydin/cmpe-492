from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from my_agent.utils.tools import tools
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

import functools
import operator

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.prebuilt import create_react_agent


from typing import List, Optional, Annotated
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph, START
from langchain_core.messages import HumanMessage, trim_messages

tavily_tool, search_tool = tools
llm = ChatOpenAI(model="gpt-4o-mini")

trimmer = trim_messages(
    max_tokens=100000,
    strategy="last",
    token_counter=llm,
    include_system=True,
)


@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    elif model_name == "anthropic":
        model =  ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    model = model.bind_tools(tools)
    return model

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


system_prompt = """Be a helpful assistant"""

# Define the function that calls the model
def call_model(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model_name = config.get('configurable', {}).get("model_name", "anthropic")
    model = _get_model(model_name)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name=name)]
    }

def create_team_supervisor(llm, system_prompt, members) -> str:
    """An LLM-based router."""
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    return (
        prompt
        | trimmer
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )


# Define the function that calls the model
def supervisor_agent(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages

    model_name = config.get('configurable', {}).get("model_name", "openai")
    llm = _get_model(model_name)

    return create_team_supervisor(
        llm,
        "You are a supervisor tasked with managing a conversation between the"
        " following workers:  Search, WebScraper. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH.",
        ["Search", "WebScraper"],
    )

def search_node(state, config):
    model_name = config.get('configurable', {}).get("model_name", "openai")
    llm = _get_model(model_name)

    def agent_node(agent, name):
        result = agent.invoke(state)
        return {
            "messages": [HumanMessage(content=result["messages"][-1].content, name=name)]
        }
    
    search_agent = create_react_agent(llm, tools=[tavily_tool])
    return functools.partial(agent_node, agent=search_agent, name="Search")

def research_node(state, config):
    model_name = config.get('configurable', {}).get("model_name", "openai")
    llm = _get_model(model_name)
    
    def agent_node(agent, name):
        result = agent.invoke(state)
        return {
            "messages": [HumanMessage(content=result["messages"][-1].content, name=name)]
        }
    research_agent = create_react_agent(llm, tools=[search_tool])
    return functools.partial(agent_node, agent=research_agent, name="WebScraper")


# Define the function to execute tools
tool_node = ToolNode(tools)