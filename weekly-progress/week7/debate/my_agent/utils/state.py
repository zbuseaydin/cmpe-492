from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    team1_finalized: Annotated[Sequence[BaseMessage], add_messages]
    team2_finalized: Annotated[Sequence[BaseMessage], add_messages]
    topic: str
    current_speaker: str
    debate_stage: str
    next: str
    jury_decision: str
