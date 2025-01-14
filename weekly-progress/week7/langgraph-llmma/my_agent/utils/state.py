from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence

# ResearchTeam graph state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # The team members are tracked so they are aware of
    # the others' skill-sets
    team_members: Sequence[str]
    # Used to route work. The supervisor calls a function
    # that will update this every time it makes a decision
    next: str
