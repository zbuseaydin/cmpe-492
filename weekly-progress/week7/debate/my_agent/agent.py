from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from my_agent.utils.nodes import host_node, team_supervisor_node, research_agent_node, antithesis_agent_node, summarizer_agent_node, jury_node
from my_agent.utils.state import AgentState


# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["openai"]


# Define a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfig)

# Create the graph
debate_graph = StateGraph(AgentState)

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
debate_graph.set_entry_point("Host")
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
graph = debate_graph.compile()
