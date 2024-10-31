from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from my_agent.utils.nodes import search_node, research_node, supervisor_agent
from my_agent.utils.state import AgentState


# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["openai", "anthropic"]


# Define a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfig)

workflow.add_node("Search", search_node)
workflow.add_node("WebScraper", research_node)
workflow.add_node("supervisor", supervisor_agent)

# Define the control flow
workflow.add_edge("Search", "supervisor")
workflow.add_edge("WebScraper", "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {"Search": "Search", "WebScraper": "WebScraper", "FINISH": END},
)

workflow.set_entry_point("supervisor")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = workflow.compile()
