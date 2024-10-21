from dotenv import load_dotenv
import operator
from typing import List, Dict, Literal, Annotated, Sequence
from typing_extensions import TypedDict
from pydantic import BaseModel
from langgraph.graph import END, StateGraph, START
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

tavily_tool = TavilySearchResults(max_results=5)
llm = ChatCohere(model="command-r-plus")

#llm = ChatOpenAI()

# Define the state types
class DebateState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    team1_finalized: Annotated[Sequence[BaseMessage], operator.add]
    team2_finalized: Annotated[Sequence[BaseMessage], operator.add]
    topic: str
    current_speaker: str
    debate_stage: str
    next: str

# Define the nodes
def host_node(state: DebateState) -> DebateState:
    current_stage = state["debate_stage"]
    change = {}
    
    if current_stage == "opening_statements":
        if state["current_speaker"] == "Team1":
            change['team1_finalized'] = [state['messages'][-1]] 
            change['next'] = "Team2Supervisor"
            change['current_speaker'] = "Team2"
        else:
            change["debate_stage"] = "rebuttal"
            change['next'] = "Team1Supervisor"
            change['current_speaker'] = "Team1"
    elif current_stage == "rebuttal":
        if state["current_speaker"] == "Team1":
            change['team1_finalized'] = [state['messages'][-1]]
            change['next'] = "Team2Supervisor"
            change['current_speaker'] = "Team2"
        else:
            change['team2_finalized'] = [state['messages'][-1]]
            change["debate_stage"] = "closing_statements"
            change['next'] = "Team1Supervisor"
            change['current_speaker'] = "Team1"
    elif current_stage == "closing_statements":
        if state["current_speaker"] == "Team1":
            change['team1_finalized'] = [state['messages'][-1]]
            change['next'] = "Team2Supervisor"
            change['current_speaker'] = "Team2"
        else:
            change['team2_finalized'] = [state['messages'][-1]]
            change["debate_stage"] = "evaluation"
            change['next'] = "Jury"
    return change

options = ["Researcher", "Antithesis Generator", "Speech Generator", "FINISH"]
class routeResponse(BaseModel):
    next: Literal["Researcher", "Antithesis Generator", "Speech Generator", "FINISH"]

sides = {"Team1": "Affirmative", "Team2": "Negative"}

def team_supervisor_node(state: DebateState) -> DebateState:
    side = sides[state['current_speaker']]
    
    # Determine the opposing team's finalized messages
    if state["current_speaker"] == "Team1":
        opposing_team_finalized = state["team2_finalized"]
    else:
        opposing_team_finalized = state["team1_finalized"]

    # Convert the finalized messages into a string for the prompt
    opposing_team_text = "\n".join([msg for msg in opposing_team_finalized])

    # Instructions for the supervisor
    agent_instructions = (
        "Each member has distinct expertise:\n"
        "- Researcher: gathers new data and information on the topic.\n"
        "- Antithesis Generator: creates questions or counters opposing arguments.\n"
        "- Speech Generator: generates speeches summarizing the team's stance.\n\n"
        "Decide which member should act next, considering the debate's current state. "
        "You should select 'FINISH' when you believe the team has completed their tasks "
        "for this stage of the debate."
    )

    # Create the prompt with the opposing team's finalized answer included
    prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            "You are the supervisor of the {side} team in the debate on {topic}. "
            "You must guide your team by selecting the appropriate member or choosing 'FINISH' when the team "
            "has completed their tasks for this stage of the debate."
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "user",
            "The debate is currently in the {stage} stage. "
            "Based on the recent discussions, decide who should act next. "
            "If you believe the team has finished their tasks for this stage, select 'FINISH'.\n\n"
            "The opposing team has finalized the following text:\n"
            "{opponent_text}\n\n"
            "Please respond only in the following format:\n"
            "{{\"next\": \"[Your selection here]\"}}\n"
            "Your choices are: [Researcher, Antithesis Generator, Speech Generator, FINISH]."
        )
    ]).partial(
        side=side, 
        topic=state["topic"], 
        stage=state["debate_stage"], 
        agent_instructions=agent_instructions, 
        opponent_text=opposing_team_text  # Include the opposing team's finalized text
    )

    supervisor_chain = prompt | llm.with_structured_output(routeResponse)
    response = supervisor_chain.invoke(state)
    return {'next': response.next}



def research_agent_node(state: DebateState, team_name: str) -> DebateState:
    search_results = tavily_tool.run(state["topic"])

    # Create a prompt that can use the state directly
    prompt = ChatPromptTemplate.from_template(
        "You are the research agent for {team_name} in the debate on {topic}. "
        "Use your expertise to provide relevant information based on the search results: {results}."
    ).partial(
        team_name=team_name,
        topic=state["topic"],
        results=search_results
    )
    
    # Create a chain between the prompt and the LLM
    chain = prompt | llm
    
    # Invoke the chain with the state
    response = chain.invoke(state)
    return {'messages' : [HumanMessage(content=response.content)]}


def antithesis_agent_node(state: DebateState, team_name: str) -> DebateState:
    prompt = ChatPromptTemplate.from_template(
        "You are the antithesis agent for {team_name} in the debate on {topic}. "
        "Generate counter-arguments or challenges based on the current discussion."
    ).partial(
        team_name=team_name,
        topic=state["topic"]
    )
    chain = prompt | llm
    # Invoke the chain with the state
    response = chain.invoke(state)
    return {'messages' : [response.content]}


def summarizer_agent_node(state: DebateState, team_name: str) -> DebateState:
    prompt = ChatPromptTemplate.from_template(
        "You are the speech generator for {team_name} in the debate on {topic}. "
        "Provide a speech summarizing your team's position based on the current discussion."
    ).partial(
        team_name=team_name,
        topic=state["topic"]
    )
    
    chain = prompt | llm
    # Invoke the chain with the state
    response = chain.invoke(state)
    return {'messages' : [response.content]}


def jury_node(state: DebateState) -> DebateState:
    # Use the team1_finalized and team2_finalized directly from the state
    team1_finalized_text = "\n".join([msg for msg in state["team1_finalized"]])
    team2_finalized_text = "\n".join([msg for msg in state["team2_finalized"]])

    # Create the prompt template with placeholders
    prompt_template = ChatPromptTemplate.from_template(
        "You are the jury evaluating a debate on the topic: {topic}. "
        "Here are the final arguments from both teams:\n\n"
        "Team 1 (Affirmative):\n{team1_finalized}\n\n"
        "Team 2 (Negative):\n{team2_finalized}\n\n"
        "Based on these arguments, please provide a verdict on which team presented a stronger case."
    ).partial(
        topic=state["topic"],
        team1_finalized=team1_finalized_text,
        team2_finalized=team2_finalized_text
    )

    # Convert the prompt into a list of BaseMessages that the LLM can process
    prompt_messages = prompt_template.format_messages()

    # Invoke the LLM with the list of messages
    response = llm.invoke(prompt_messages)

    return {'messages': [HumanMessage(content=response.content)]}



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
    messages=[HumanMessage(content="Let's start the debate on the topic: AI's impact on job markets")],
    topic="AI's impact on job markets",  
    current_speaker="Team2",
    debate_stage="opening_statements",
    next="Host"
)

# Run the debate
final_state = debate_chain.invoke(initial_state)

# Print the debate results
print("Debate Results:")
for message in final_state["messages"]:
    if 'content' in message:
        print(f"{message.content}\n")
    else:
        print(message)
