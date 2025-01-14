import time
from datetime import datetime
import csv
import os
import asyncio
from typing import List, Dict
from single_agent import SingleAgent

class VotingAgents:
    def __init__(self, config):
        self.config = config
        self.agents = []
        
        # Create agents based on individual configurations
        for agent_config in config.get('agents', []):
            self.agents.append(SingleAgent(agent_config))
    
    def _make_final_decision(self, responses: List[Dict], response_ids: List[int]) -> Dict:
        # Count votes for each decision
        votes = {'LEFT': 0, 'RIGHT': 0}
        
        for response in responses:
            if response['decision'] in votes:
                votes[response['decision']] += 1
        
        # Determine majority decision
        final_decision = 'LEFT' if votes['LEFT'] >= votes['RIGHT'] else 'RIGHT'
        
        # Create reference to individual agent responses
        consensus_reason = (
            f"Consensus Decision ({votes[final_decision]} out of {len(responses)} agents). "
            f"Vote split: {votes['LEFT']} for LEFT, {votes['RIGHT']} for RIGHT. "
            f"See individual agent responses in rows: {', '.join(map(str, response_ids))}"
        )
        
        return {
            'decision': final_decision,
            'reason': consensus_reason
        }
    
    async def analyze(self, scenario, start_time):
        # Collect responses from all agents
        responses = []
        response_ids = []
        
        # Get responses from each agent
        for agent in self.agents:
            agent_start_time = time.time()
            response = await agent.analyze(scenario, agent_start_time)
            responses.append(response)
            response_ids.append(response['csv_id'])  # Get ID from the response
        
        # Make final decision based on majority vote
        final_decision = self._make_final_decision(responses, response_ids)
        
        # Add runtime
        runtime = round(time.time() - start_time, 4)
        final_decision['runtime'] = f"{runtime}s"
        
        # Save to CSV
        self._save_to_csv(scenario, final_decision, runtime, len(responses))
        
        # Return final decision and individual responses
        return {
            **final_decision,
            'individual_responses': responses
        }
    
    def _save_to_csv(self, scenario, response, runtime, num_responses):
        csv_file = 'multi_agent_responses.csv'
        file_exists = os.path.isfile(csv_file)
        
        # Create a string representation of agent configurations
        agent_configs = [
            f"Agent{i+1}(model={agent.config['llm']['model']} temp={agent.config['llm']['temperature']})" 
            for i, agent in enumerate(self.agents)
        ]
        agent_configs_str = ' '.join(agent_configs)
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'num_agents', 'agent_configs', 'scenario_type', 
                               'legal_status', 'left_group', 'right_group', 
                               'decision', 'reason', 'runtime'])
            
            writer.writerow([
                datetime.now().isoformat(),
                num_responses,
                agent_configs_str,  # Add agent configurations
                scenario.type,
                scenario.legalStatus,
                SingleAgent.format_group_csv(scenario.left),
                SingleAgent.format_group_csv(scenario.right),
                response['decision'],
                response['reason'],
                runtime
            ])
