from typing import Dict, List, Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
from datetime import datetime
import time
from langgraph.graph import Graph, StateGraph

class AgentState(TypedDict):
    """The state of the agent system."""
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    next: str
    round: int
    decisions: Dict[str, List[Dict]]
    final_decision: Dict

def create_agent(config: Dict, name: str):
    """Create an agent with specific configuration."""
    llm = ChatOpenAI(
        model=config['llm']['model'],
        temperature=config['llm']['temperature']
    )
    
    # Initial decision prompt
    initial_prompt = ChatPromptTemplate.from_template("""
You are an AI agent participating in a moral decision-making experiment. Your attributes are:
{agent_attributes}

You need to decide which group to save in this scenario:
Left Group: {left_desc}
Right Group: {right_desc}

Provide your decision in JSON format:
{
    "decision": "LEFT" or "RIGHT",
    "reason": "Your reasoning considering your attributes"
}
""")

    # Discussion prompt for subsequent rounds
    discussion_prompt = ChatPromptTemplate.from_template("""
Previous round decisions:
{previous_decisions}

Based on these decisions and reasoning, would you like to change your decision or defend your previous choice?

Provide your decision in JSON format:
{{
    "decision": "LEFT" or "RIGHT",
    "reason": "Your reasoning considering your attributes and the discussion"
}}
""")

    def format_agent_attributes(attributes):
        return "\n".join([f"- {k}: {v}" for k, v in attributes.items()])

    def make_decision(state):
        messages = state['messages']
        round_num = state['round']
        
        if round_num == 1:
            # First round decision
            response = initial_prompt | llm | StrOutputParser()
            result = response.invoke({
                "agent_attributes": format_agent_attributes(config['attributes']),
                "left_desc": messages[0].content['left_desc'],
                "right_desc": messages[0].content['right_desc']
            })
        else:
            # Subsequent rounds with discussion
            previous_decisions = json.dumps(state['decisions'], indent=2)
            response = discussion_prompt | llm | StrOutputParser()
            result = response.invoke({
                "previous_decisions": previous_decisions
            })

        return json.loads(result)

    return make_decision

def create_moral_machine_graph(agents_config: List[Dict]) -> Graph:
    """Create the multi-agent decision making graph."""
    
    # Create agents
    agents = {
        f"agent_{i}": create_agent(config, f"agent_{i}")
        for i, config in enumerate(agents_config)
    }

    def should_continue(state):
        """Determine if another round is needed."""
        if state['round'] >= 3:
            return "end"
            
        decisions = [d['decision'] for d in state['decisions'].values()]
        if len(set(decisions)) == 1:
            return "end"
            
        return "continue"

    def aggregate_decisions(state):
        """Aggregate final decisions and determine the outcome."""
        all_decisions = []
        for agent_decisions in state['decisions'].values():
            all_decisions.append(agent_decisions[-1]['decision'])
            
        most_common = max(set(all_decisions), key=all_decisions.count)
        supporting_reasons = [
            d[-1]['reason'] 
            for d in state['decisions'].values() 
            if d[-1]['decision'] == most_common
        ]
        
        state['final_decision'] = {
            "decision": most_common,
            "supporting_reasons": supporting_reasons,
            "round_count": state['round']
        }
        return state

    # Build the graph
    workflow = StateGraph(AgentState)

    # Add agent nodes
    for agent_name, agent_fn in agents.items():
        workflow.add_node(agent_name, agent_fn)

    # Add decision aggregator
    workflow.add_node("aggregate", aggregate_decisions)

    # Connect nodes
    for agent_name in agents:
        workflow.add_edge(agent_name, should_continue)
    
    workflow.add_edge("continue", list(agents.keys())[0])
    workflow.add_edge("end", "aggregate")

    # Set entry point
    workflow.set_entry_point("agent_0")

    return workflow.compile()

def run_moral_machine_experiment(scenario: Dict, agents_config: List[Dict]) -> Dict:
    """Run a moral machine experiment with multiple agents."""
    graph = create_moral_machine_graph(agents_config)
    
    initial_state = {
        "messages": [HumanMessage(content=scenario)],
        "next": "agent_0",
        "round": 1,
        "decisions": {},
        "final_decision": {}
    }
    
    result = graph.invoke(initial_state)
    return result['final_decision']

class MultiAgentSystem:
    def __init__(self, agents, config, prompt_template):
        """
        Initialize the MultiAgentSystem with agents and their attributes.
        
        Args:
            agents: List of dictionaries containing agent attributes
            config: Configuration dictionary with LLM and system settings
            prompt_template: Name of the prompt template to use
        """
        self.agents = agents
        self.config = config
        self.prompt_template = prompt_template
        self.setup_agents()

    def setup_agents(self):
        self.llm = ChatOpenAI(**self.config['llm'])
        self.initial_prompt = ChatPromptTemplate.from_template(
            self.config['prompt_templates'][self.prompt_template]
        )
        self.discussion_prompt = ChatPromptTemplate.from_template("""
Previous round decisions and reasoning:
{previous_decisions}

Based on the above discussion, would you like to change your decision or defend your previous choice?
Consider your attributes:
- Gender: {agent_gender}
- Role: {agent_role}
- Age: {agent_age}
- Religious Orientation: {agent_religious_orientation} (0 = Atheist, 1 = Highly Religious)
- Political Orientation: {agent_political_orientation} (0 = Highly Conservative, 1 = Highly Progressive)

Provide your decision in JSON format:
{{
    "decision": "LEFT" or "RIGHT",
    "reason": "Your reasoning considering your attributes and the discussion"
}}
""")

    async def run_debate(self, scenario_data):
        current_round = 1
        decisions_history = []
        
        while current_round <= self.config['multiagent']['max_rounds']:
            print(f'{current_round=}')
            round_decisions = {}
            
            # Get decisions from all agents
            for i, agent in enumerate(self.agents):
                if current_round == 1:
                    decision = await self.get_initial_decision(agent, scenario_data)
                else:
                    decision = await self.get_discussion_decision(
                        agent, 
                        decisions_history[-1]
                    )
                round_decisions[f"agent_{i}"] = decision
            
            decisions_history.append(round_decisions)
            
            # Check for consensus
            decisions = [d['decision'] for d in round_decisions.values()]
            if len(set(decisions)) == 1 or current_round == self.config['multiagent']['max_rounds']:
                most_common = max(set(decisions), key=decisions.count)
                supporting_reasons = [
                    d['reason'] 
                    for d in round_decisions.values() 
                    if d['decision'] == most_common
                ]
                
                return {
                    "decision": most_common,
                    "supporting_reasons": supporting_reasons,
                    "round_count": current_round,
                    "debate_history": decisions_history
                }
            
            current_round += 1

    async def get_initial_decision(self, agent, scenario_data):
        chain = self.initial_prompt | self.llm | StrOutputParser()
        result = await chain.ainvoke({
            "agent_gender": agent['attributes']['gender'],
            "agent_role": agent['attributes']['role'],
            "agent_age": agent['attributes']['age'],
            "agent_religious_orientation": agent['attributes']['religious_orientation'],
            "agent_political_orientation": agent['attributes']['political_orientation'],
            **scenario_data
        })
        # Clean the result of any markdown formatting
        result = result.strip().replace('```json', '').replace('```', '')
        return json.loads(result)

    async def get_discussion_decision(self, agent, previous_round):
        chain = self.discussion_prompt | self.llm | StrOutputParser()
        result = await chain.ainvoke({
            "previous_decisions": json.dumps(previous_round, indent=2),
            "agent_gender": agent['attributes']['gender'],
            "agent_role": agent['attributes']['role'],
            "agent_age": agent['attributes']['age'],
            "agent_religious_orientation": agent['attributes']['religious_orientation'],
            "agent_political_orientation": agent['attributes']['political_orientation']
        })
        # Clean the result of any markdown formatting
        result = result.strip().replace('```json', '').replace('```', '')
        return json.loads(result)
