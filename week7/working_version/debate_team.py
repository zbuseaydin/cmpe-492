from dotenv import load_dotenv
import operator
from typing import Literal, Annotated, Sequence
from typing_extensions import TypedDict
from pydantic import BaseModel
from langgraph.graph import END, StateGraph, START
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from debate_team_variables import prompts, llms

load_dotenv()


host_system_prompt = prompts["host"]["system"][0]
# supervisor_agent_instructions = prompts["team_supervisor"]["agent_instructions"][0]
# supervisor_system_prompt = prompts["team_supervisor"]["system"][3]
# supervisor_user_prompt = prompts["team_supervisor"]["user"][2]
research_prompt = prompts['research_agent'][0]
antithesis_prompt = prompts['antithesis_agent'][1]
summarizer_prompt = prompts['summarizer_agent'][0]
jury_prompt = prompts['jury'][0]
topic_statement, topic = prompts['topic_statement'][0]

host_llm_name = [*llms][1]
host_model = llms[host_llm_name][1]
team1_llm_name = [*llms][1]
team1_model = llms[team1_llm_name][1]
team2_llm_name = [*llms][1]
team2_model = llms[team2_llm_name][1]
jury_llm_name = [*llms][1]
jury_model = llms[jury_llm_name][1]


if host_llm_name == 'cohere':
    host_llm = ChatCohere(model=host_model)
elif host_llm_name == 'openai':
    host_llm = ChatOpenAI(model=host_model)
else:
    host_llm = ChatCohere(model="command-r-plus")

if team1_llm_name == 'cohere':
    team1_llm = ChatCohere(model=team1_model)
elif team1_llm_name == 'openai':
    team1_llm = ChatOpenAI(model=team1_model)
else:
    team1_llm = ChatCohere(model="command-r-plus")

if team2_llm_name == 'cohere':
    team2_llm = ChatCohere(model=team2_model)
elif team2_llm_name == 'openai':
    team2_llm = ChatOpenAI(model=team2_model)
else:
    team2_llm = ChatCohere(model="command-r-plus")

if jury_llm_name == 'cohere':
    jury_llm = ChatCohere(model=jury_model)
elif jury_llm_name == 'openai':
    jury_llm = ChatOpenAI(model=jury_model)
else:
    jury_llm = ChatCohere(model="command-r-plus")


sides = {"Team1": "Affirmative", "Team2": "Negative"}


class routeResponse(BaseModel):
    next: Literal["Researcher", "Antithesis Generator", "Speech Generator", "FINISH"]


# Define the state types
class DebateState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    team1_finalized: Annotated[Sequence[BaseMessage], operator.add]
    team2_finalized: Annotated[Sequence[BaseMessage], operator.add]
    host_messages: Annotated[Sequence[BaseMessage], operator.add]
    topic: str
    current_speaker: str
    debate_stage: str
    next: str
    jury_decision: str



# Define the nodes
def host_node(state: DebateState) -> DebateState:
    current_stage = state["debate_stage"]
    change = {}
    last_message = None
    if len(state['messages']) > 2 and state['messages'][-2] != "Passed to the next agent.":
        last_message = state['messages'][-2]
    
    if current_stage == "opening_statements":
        if state["current_speaker"] == "Team1":
            if last_message:
                change['team1_finalized'] = [last_message]
            change['next'] = "Team2Supervisor"
            change['current_speaker'] = "Team2"
        else:
            change["debate_stage"] = "rebuttal"
            change['next'] = "Team1Supervisor"
            change['current_speaker'] = "Team1"
    elif current_stage == "rebuttal":
        if state["current_speaker"] == "Team1":
            if last_message:
                change['team1_finalized'] = [last_message]
            change['next'] = "Team2Supervisor"
            change['current_speaker'] = "Team2"
        else:
            if last_message:
                change['team2_finalized'] = [last_message]
            change["debate_stage"] = "closing_statements"
            change['next'] = "Team1Supervisor"
            change['current_speaker'] = "Team1"
    elif current_stage == "closing_statements":
        if state["current_speaker"] == "Team1":
            if last_message:
                change['team1_finalized'] = [last_message]
            change['next'] = "Team2Supervisor"
            change['current_speaker'] = "Team2"
        else:
            if last_message:
                change['team2_finalized'] = [last_message]
            change["debate_stage"] = "evaluation"
            change['next'] = "Jury"

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            host_system_prompt
        )
    ]).partial(
        side1="Affirmative",
        side2="Negative",
        topic=state["topic"],
    )

    host_chain = prompt | host_llm
    
    response = host_chain.invoke(change)
    change["host_messages"] = [response.content]
    return change


def team_supervisor_node(state: DebateState) -> DebateState:
    
    # Determine the opposing team's finalized messages
    if state["current_speaker"] == "Team1":
        llm = team1_llm
    else:
        llm = team2_llm

    # Create the prompt with the opposing team's finalized answer included
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            state["host_messages"][-1]
        )
    ])

    supervisor_chain = prompt | llm.with_structured_output(routeResponse)
    response = supervisor_chain.invoke(state)
    return {
        'next': response.next,
        'messages': ['Passed to the next agent.']
        }


def research_agent_node(state: DebateState, team_name: str) -> DebateState:
    search_results = tavily_tool.run(state["topic"])

    # Create a prompt that can use the state directly
    prompt = ChatPromptTemplate.from_template(
        research_prompt
    ).partial(
        team_name=team_name,
        topic=state["topic"],
        results=search_results
    )
    prompt_messages = prompt.format_messages()
    response = team1_llm.invoke(prompt_messages) if state["current_speaker"] == "Team1" else team2_llm.invoke(prompt_messages)
    return {
        'messages' : [HumanMessage(content=response.content)], 
        "host_messages" : [state["host_messages"][-1] + "\nResearcher agent is done."]
    }


def antithesis_agent_node(state: DebateState, team_name: str) -> DebateState:
    global antithesis_prompt
    side = sides[state['current_speaker']]
    if state["current_speaker"] == "Team1":
        opposing_team_finalized = state["team2_finalized"]
        current_team_finalized = state["team1_finalized"]
        llm = team1_llm
    else:
        opposing_team_finalized = state["team1_finalized"]
        current_team_finalized = state["team2_finalized"]
        llm = team2_llm
    if opposing_team_finalized:
        antithesis_prompt += "The opposing team has finalized the following text:\n{opponent_text}\n\n"
    if current_team_finalized:
        antithesis_prompt += "Your team has finalized the following text:\n{current_team_text}\n\n"
    prompt = ChatPromptTemplate.from_template(
        antithesis_prompt
    ).partial(
        team_name=team_name,
        topic=state["topic"],
        opponent_text=opposing_team_finalized,
        current_team_text=current_team_finalized,
        side=side
    )
#    chain = prompt | llm
    # Invoke the chain with the state
    prompt_messages = prompt.format_messages()
    response = llm.invoke(prompt_messages)
    return {'messages' : [HumanMessage(content=response.content)], "host_messages" : [state["host_messages"][-1] + "\nAntithesis agent is done."]}


def summarizer_agent_node(state: DebateState, team_name: str) -> DebateState:
    prompt = ChatPromptTemplate.from_template(
        summarizer_prompt
    ).partial(
        team_name=team_name,
        topic=state["topic"]
    )
    chain = prompt | team1_llm if state["current_speaker"] == "Team1" else prompt | team2_llm
    # Invoke the chain with the state
    response = chain.invoke(state)
    return {'messages' : [HumanMessage(content=response.content)], "host_messages" : [state["host_messages"][-1] + "\nSummarizer agent is done."]}


def jury_node(state: DebateState) -> DebateState:
    # Use the team1_finalized and team2_finalized directly from the state
    team1_finalized_text = "\n".join([str(msg) for msg in state["team1_finalized"]])
    team2_finalized_text = "\n".join([str(msg) for msg in state["team2_finalized"]])

    # Create the prompt template with placeholders
    prompt_template = ChatPromptTemplate.from_template(
        jury_prompt
    ).partial(
        topic=state["topic"],
        team1_finalized=team1_finalized_text,
        team2_finalized=team2_finalized_text
    )
    # Convert the prompt into a list of BaseMessages that the LLM can process
    prompt_messages = prompt_template.format_messages()
    response = jury_llm.invoke(prompt_messages)
    return {'jury_decision': response.content}


tavily_tool = TavilySearchResults(max_results=5)

# Create the graph
debate_graph = StateGraph(DebateState)

# Add nodes
debate_graph.add_node("Host", host_node)
debate_graph.add_node("Team1Supervisor", team_supervisor_node)
debate_graph.add_node("Team2Supervisor", team_supervisor_node)

# Add team-specific agent nodes
for team in ["Team1", "Team2"]:
    debate_graph.add_node(f"{team}ResearchAgent", lambda s: research_agent_node(s, team))
    debate_graph.add_node(f"{team}AntithesisAgent", lambda s: antithesis_agent_node(s, team))
    debate_graph.add_node(f"{team}SummarizerAgent", lambda s: summarizer_agent_node(s, team))

debate_graph.add_node("Jury", jury_node)

# Add edges
debate_graph.add_edge(START, "Host")
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

for team in ["Team1", "Team2"]:
    supervisor = f"{team}Supervisor"

    debate_graph.add_conditional_edges(
        supervisor,
        lambda x: x["next"],
        {
            "Researcher": f"{team}ResearchAgent",
            "Antithesis Generator": f"{team}AntithesisAgent",
            "Speech Generator": f"{team}SummarizerAgent",
            "FINISH": "Host"
        }
    )
    debate_graph.add_edge(f"{team}ResearchAgent", supervisor)
    debate_graph.add_edge(f"{team}AntithesisAgent", supervisor)
    debate_graph.add_edge(f"{team}SummarizerAgent", supervisor)

debate_graph.add_edge("Jury", END)

# Compile the graph
debate_chain = debate_graph.compile()

# Initialize the debate
initial_state = DebateState(
    messages=[HumanMessage(content=topic_statement)],
    topic=topic,  
    current_speaker="Team2",
    debate_stage="opening_statements",
    next="Host"
)

# Run the debate
final_state = debate_chain.invoke(initial_state, {"recursion_limit": 50})

# Print the debate results
print("Debate Results:")
for message in final_state["messages"]:
    if 'content' in message:
        print(f"{message.content}\n")
    else:
        print(message)