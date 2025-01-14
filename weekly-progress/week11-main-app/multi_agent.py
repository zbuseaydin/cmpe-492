from typing import Dict, List, Tuple, Any
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolInvocation
import json
import time

class DebateState(TypedDict):
    messages: List[Dict]
    scenario: Dict

class MultiAgents:
    def __init__(self, config):
        self.agents = []
        self.setup_agents(config["agents"])
        self.graph = self.create_debate_graph()

    def setup_agents(self, agent_configs):
        """Initialize agents with their specific configurations"""
        for config in agent_configs:
            agent = {
                "id": config["agent_id"],
                "llm": ChatOpenAI(
                    model=config["llm"]["model"],
                    temperature=config["llm"]["temperature"]
                ),
                "prompt": config["prompt_template"],
                "attributes": config["attributes"]
            }
            self.agents.append(agent)

    def format_agent_message(self, scenario_data: Dict) -> str:
        """Format the scenario data into a message for the agents"""
        return (
            f"Scenario Type: {scenario_data['scenario_type']}\n"
            f"Legal Status: {scenario_data['legal_status']}\n"
            f"Left Path: {scenario_data['left']}\n"
            f"Right Path: {scenario_data['right']}\n"
        )

    def agent_node(self, agent_id: int):
        """Create a node function for a specific agent"""
        def node_function(state):
            messages = state["messages"]
            scenario = state["scenario"]
            
            agent = next(a for a in self.agents if a["id"] == agent_id)
            
            # Only include messages from previous rounds
            current_round = len(messages) // 3
            previous_messages = messages[:current_round * 3]
            
            # Format the input for the agent
            current_context = "\n".join([
                f"Previous discussion:" if previous_messages else "",
                *[f"Agent {m['agent']}: {m['content']}" for m in previous_messages],
                "\nBased on the scenario and previous discussion, provide your decision and reasoning:"
            ])
            
            print(scenario)
            # Calculate group sizes and format left and right descriptions
            left_group_size = scenario['left']['group_size']
            right_group_size = scenario['right']['group_size']
            
            left_desc = json.dumps({
                "total number of fatalities": left_group_size,
                "members": [f"{v} {k}{'s' if v > 1 else ''}" for k, v in scenario['left']['members'].items()]
            }, indent=4)
            
            right_desc = json.dumps({
                "total number of fatalities": right_group_size,
                "members": [f"{v} {k}{'s' if v > 1 else ''}" for k, v in scenario['right']['members'].items()]
            }, indent=4)
            

            prompt = agent["prompt"].format(
                scenario_type=scenario['type'],
                legal_status=scenario['legalStatus'],
                left_desc=left_desc,
                right_desc=right_desc,
                agent_role = agent["attributes"]["role"],
                agent_gender = agent["attributes"]["gender"],
                agent_age = agent["attributes"]["age"],
                agent_education_level = agent["attributes"]["education_level"],
                agent_calmness = agent["attributes"]["calmness"],
                agent_empathy = agent["attributes"]["empathy"],
                agent_analytical_thinking = agent["attributes"]["analytical_thinking"],
                agent_risk_tolerance = agent["attributes"]["risk_tolerance"],
                agent_decisiveness = agent["attributes"]["decisiveness"]
            )
            
            response = agent["llm"].invoke(
                [HumanMessage(content=f"{prompt}\n\n{current_context}")]
            )
            
            try:
                # Strip markdown code block if present
                content = response.content.strip("```json").strip()
                # Remove any newlines and other markdown artifacts
                content = content.replace("\\n", '').replace("\n", '').replace("```", '')
                print(f'{content=}')
                # Parse the JSON content
                decision_data = json.loads(content)
                decision = decision_data.get("decision", "").upper()
                reason = decision_data.get("reason", "")
            except json.JSONDecodeError:
                content_lower = content.lower().replace('"', '')
                print(f'{content_lower=}')
                if "decision: left" in content_lower:
                    decision = "LEFT"
                elif "decision: right" in content_lower:
                    decision = "RIGHT"
                else:
                    decision = "UNKNOWN"
                
                # Parse reason similarly
                if "reason:" in content_lower:
                    reason = content_lower.split("reason:")[1].strip()
                else:
                    reason = "UNKNOWN"

            messages.append({
                "agent": agent_id,
                "content": f"Decision: {decision}\nReason: {reason}"
            })
            
            # Ensure the state is updated
            return {"messages": messages, "scenario": scenario, "next": "continue"}
        
        return node_function

    def host_node(self, state):
        """Determine the next agent or end the debate"""
        if len(state["messages"]) >= 9:  # Maximum 3 rounds
            return {"messages": state["messages"], "scenario": state["scenario"], "next": END}
            
        # Check for consensus
        recent_decisions = []
        for msg in state["messages"][-3:]:
            content_lower = msg["content"].lower()
            if "decision: left" in content_lower:
                recent_decisions.append("LEFT")
            elif "decision: right" in content_lower:
                recent_decisions.append("RIGHT")
        
        if len(recent_decisions) == 3:
            if all(d == recent_decisions[0] for d in recent_decisions) or len(state["messages"]) >= 9:
                return {"messages": state["messages"], "scenario": state["scenario"], "next": END}
        
        # Determine next agent
        current_round = len(state["messages"]) % 3
        next_agent = f"agent{current_round + 1}"
        return {"messages": state["messages"], "scenario": state["scenario"], "next": next_agent}

    def create_debate_graph(self):
        """Create the LangGraph network for agent debate"""
        workflow = StateGraph(DebateState)
        
        # Add nodes
        workflow.add_node("host", self.host_node)
        for i in range(1, 4):
            workflow.add_node(f"agent{i}", self.agent_node(i))
        
        # Add edges
        workflow.add_edge(START, "host")
        
        # Add conditional edges from host to agents or END
        workflow.add_conditional_edges(
            "host",
            lambda x: x["next"],
            {
                "agent1": "agent1",
                "agent2": "agent2",
                "agent3": "agent3",
                END: END
            }
        )
        
        # Add edges from agents back to host
        for i in range(1, 4):
            workflow.add_edge(f"agent{i}", "host")
        
        return workflow.compile()

    def get_final_decision(self, messages: List[Dict]) -> Dict:
        """Determine final decision based on majority vote"""
        decisions = {"LEFT": 0, "RIGHT": 0}
        reasons = []
        
        for msg in messages:
            content_lower = msg["content"].lower()
            if "decision: left" in content_lower:
                decisions["LEFT"] += 1
            elif "decision: right" in content_lower:
                decisions["RIGHT"] += 1
            if "reason:" in msg["content"]:
                reasons.append(msg["content"].split("Reason:")[1].strip())
        
        final_decision = "LEFT" if decisions["LEFT"] >= decisions["RIGHT"] else "RIGHT"
        return {
            "decision": final_decision,
            "vote_count": decisions,
            "reasons": reasons,
            "consensus": decisions["LEFT"] == 3 or decisions["RIGHT"] == 3
        }

    async def analyze(self, scenario: Dict, start_time: float) -> Dict:
        """Analyze the scenario through agent debate and return results"""
        initial_state = {
            "messages": [],
            "scenario": scenario
        }
        
        # Run the debate
        final_state = self.graph.invoke(initial_state)
        
        # Get final decision and metadata
        result = self.get_final_decision(final_state["messages"])
        
        # Calculate runtime
        total_runtime = round(time.time() - start_time, 4)
        
        return {
            "decision": result["decision"],
            "vote_count": result["vote_count"],
            "reasons": result["reasons"],
            "consensus": result["consensus"],
            "debate_rounds": len(final_state["messages"]) // 3,
            "runtime": f"{total_runtime}s"
        }
