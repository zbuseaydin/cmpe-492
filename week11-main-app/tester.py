import json
import pandas as pd
import matplotlib.pyplot as plt
from single_agent import SingleAgent
from multi_agent import MultiAgents
import asyncio
import time

# Human results for comparison (example data - replace with actual data)
HUMAN_RESULTS = {
    "Fit vs Fat Comparison": 0.65,  # 65% chose left (fit) option
    "Young vs Old Comparison": 0.45,  # 45% chose left (young) option
    "Male vs Female Comparison": 0.48,  # 48% chose left (male) option
    "Social Status Comparison": 0.55,  # 55% chose left (high status) option
    "Animal vs Person Comparison": 0.82,  # 82% chose left (human) option
    "Few vs More Comparison": 0.25,  # 25% chose left (few) option
}

class ScenarioTester:
    def __init__(self):
        self.load_config()
        self.load_scenarios()
        self.results = []

    def load_config(self):
        with open('config.json', 'r') as f:
            self.config = json.load(f)

    def load_scenarios(self):
        with open('scenarios.json', 'r') as f:
            self.scenarios = json.load(f)

    async def run_scenario(self, scenario, agent):
        # Prepare scenario in the format expected by the agent
        scenario_data = {
            "type": scenario["type"],
            "legalStatus": scenario["legalStatus"],
            "left": {
                "group_size": sum(scenario["left"].values()),
                "members": scenario["left"]
            },
            "right": {
                "group_size": sum(scenario["right"].values()),
                "members": scenario["right"]
            }
        }

        # Get agent's decision
        start_time = time.time()
        result = await agent.analyze(scenario_data, start_time)
        return result

    async def test_all_scenarios(self, num_runs=10):
        # Initialize agents
        single_agent = SingleAgent(self.config['agents'][0])
        multi_agents = MultiAgents(self.config)
        
        # Dictionary to store accumulated results
        accumulated_results = []
        
        # Run tests multiple times
        for run in range(num_runs):
            run_results = []
            for scenario in self.scenarios['scenarios']:
                # Test with each agent architecture
                single_result = await self.run_scenario(scenario, single_agent)
                multi_result = await self.run_scenario(scenario, multi_agents)
                
                # Record results for this run
                run_results.append({
                    'scenario_name': scenario['name'],
                    'single_agent_decision': 1 if single_result['decision'] == 'LEFT' else 0,
                    'multi_agents_decision': 1 if multi_result['decision'] == 'LEFT' else 0,
                    'human_result': HUMAN_RESULTS[scenario['name']]
                })
            accumulated_results.append(run_results)
        
        # Calculate averages
        self.results = []
        for scenario_idx in range(len(self.scenarios['scenarios'])):
            scenario_name = self.scenarios['scenarios'][scenario_idx]['name']
            single_agent_avg = sum(run[scenario_idx]['single_agent_decision'] 
                                 for run in accumulated_results) / num_runs
            multi_agents_avg = sum(run[scenario_idx]['multi_agents_decision'] 
                                 for run in accumulated_results) / num_runs
            human_result = HUMAN_RESULTS[scenario_name]
            
            self.results.append({
                'scenario_name': scenario_name,
                'single_agent_decision': single_agent_avg,
                'multi_agents_decision': multi_agents_avg,
                'human_result': human_result
            })

    def save_results(self):
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save to CSV with average indicator
        df.to_csv('scenario_results_avg_10runs.csv', index=False)
        return df

    def plot_results(self, df):
        # Prepare data for plotting
        scenarios = df['scenario_name']
        single_agent_results = df['single_agent_decision']
        multi_results = df['multi_agents_decision']
        human_results = df['human_result']

        # Create bar plot
        x = range(len(scenarios))
        width = 0.25

        plt.figure(figsize=(15, 8))
        plt.bar([i - width for i in x], single_agent_results, width, label='Single Agent', color='blue')
        plt.bar([i for i in x], multi_results, width, label='Multi Agents', color='red')
        plt.bar([i + width for i in x], human_results, width, label='Human Results', color='gray')

        plt.xlabel('Scenarios')
        plt.ylabel('Proportion choosing LEFT option')
        plt.title('Comparison of Agent Architectures with Human Results')
        plt.xticks([i for i in x], scenarios, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig('results_comparison.png')
        plt.close()

async def main():
    tester = ScenarioTester()
    await tester.test_all_scenarios()
    results_df = tester.save_results()
    tester.plot_results(results_df)

if __name__ == "__main__":
    asyncio.run(main())
