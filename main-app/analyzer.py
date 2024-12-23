import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import os

# Function to load and process experiment results
def process_experiment_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    scenario_decisions = {}
    
    for result in data['results']:
        scenario_name = result['scenario']['name']
        attribute_level = result['scenario']['attributeLevel']
        decision = result['result']['decision']
        
        if scenario_name not in scenario_decisions:
            scenario_decisions[scenario_name] = {'LEFT': 0, 'RIGHT': 0, 'total': 0}
        
        scenario_decisions[scenario_name]['total'] += 1
        scenario_decisions[scenario_name][decision] += 1
    
    return scenario_decisions

# Real person responses data
real_responses = pd.DataFrame({
    'ScenarioType': ['Age', 'Age', 'Fitness', 'Fitness', 'Gender', 'Gender', 
                     'Social Status', 'Social Status', 'Species', 'Species', 
                     'Utilitarian', 'Utilitarian'],
    'AttributeLevel': ['Old', 'Young', 'Fat', 'Fit', 'Female', 'Male', 
                      'High', 'Low', 'Hoomans', 'Pets', 'Less', 'More'],
    'Count': [184013, 1152568, 400364, 843693, 880626, 496836, 
              120178, 47216, 1151202, 233264, 172588, 1236655],
    'Percentage': [13.767441, 86.232559, 32.182127, 67.817873, 63.931056, 36.068944,
                  71.793493, 28.206507, 83.151338, 16.848662, 12.246859, 87.753141]
})

# Load experiment results
experiment_files = [
    'experiments/age_experiment_results_20241222_144747.json',
    'experiments/role_experiment_results_20241222_145530.json',
    'experiments/gender_experiment_results_20241222_145628.json',
    'experiments/political_experiment_results_20241222_143929.json',
    'experiments/religious_experiment_results_20241222_144216.json',
    'experiments/empathy_experiment_results_20241222_145245.json',
    'experiments/education_experiment_results_20241222_144457.json'
]

# Process all experiment files
all_agent_decisions = {}
for filepath in experiment_files:
    decisions = process_experiment_file(filepath)
    for scenario, counts in decisions.items():
        if scenario not in all_agent_decisions:
            all_agent_decisions[scenario] = {'LEFT': 0, 'RIGHT': 0, 'total': 0}
        all_agent_decisions[scenario]['LEFT'] += counts['LEFT']
        all_agent_decisions[scenario]['RIGHT'] += counts['RIGHT']
        all_agent_decisions[scenario]['total'] += counts['total']

# Create mapping between scenario names and types, and their choices
scenario_info = {
    'Young vs Old Comparison': {'type': 'Age', 'left': 'Young', 'right': 'Old'},
    'Fit vs Fat Comparison': {'type': 'Fitness', 'left': 'Fit', 'right': 'Fat'},
    'Male vs Female Comparison': {'type': 'Gender', 'left': 'Male', 'right': 'Female'},
    'High vs Low Status Comparison': {'type': 'Social Status', 'left': 'High', 'right': 'Low'},
    'Human vs Pet Comparison': {'type': 'Species', 'left': 'Human', 'right': 'Pet'},
    'Few vs More Comparison': {'type': 'Utilitarian', 'left': 'Less', 'right': 'More'}
}

# Create mapping between scenario names and their types
scenario_type_mapping = {name: info['type'] for name, info in scenario_info.items()}

# Create a single figure for all scenarios
plt.figure(figsize=(15, 8))

# Set up positions for bars
scenarios = list(scenario_type_mapping.values())
x = np.arange(len(scenarios))
width = 0.35  # Width of the bars

# Prepare data for plotting
human_left_pcts = []
human_right_pcts = []
agent_left_pcts = []
agent_right_pcts = []

for scenario_name, scenario_type in scenario_type_mapping.items():
    # Get real responses data
    real_data = real_responses[real_responses['ScenarioType'] == scenario_type]
    human_left_pcts.append(real_data['Percentage'].values[0])
    human_right_pcts.append(real_data['Percentage'].values[1])
    
    # Get agent responses data
    if scenario_name in all_agent_decisions:
        agent_data = all_agent_decisions[scenario_name]
        agent_left_pct = (agent_data['LEFT'] / agent_data['total']) * 100
        agent_right_pct = (agent_data['RIGHT'] / agent_data['total']) * 100
    else:
        agent_left_pct = agent_right_pct = 0
    
    agent_left_pcts.append(agent_left_pct)
    agent_right_pcts.append(agent_right_pct)

# Create bars
plt.bar(x - width/2, human_left_pcts, width, label=f'Human - Left Choice', color='lightcoral')
plt.bar(x + width/2, agent_left_pcts, width, label=f'Agent - Left Choice', color='indianred')
plt.bar(x - width/2, human_right_pcts, width, bottom=human_left_pcts, label=f'Human - Right Choice', color='skyblue')
plt.bar(x + width/2, agent_right_pcts, width, bottom=agent_left_pcts, label=f'Agent - Right Choice', color='steelblue')

# Customize the plot
plt.xlabel('Scenario Types')
plt.ylabel('Percentage')
plt.title('Human vs Agent Responses Across Scenarios')

# Create descriptive labels for x-axis
x_labels = []
for scenario_name in scenario_type_mapping.keys():
    info = scenario_info[scenario_name]
    x_labels.append(f"{info['type']}\n{info['left']}←  →{info['right']}")

plt.xticks(x, x_labels, rotation=45, ha='right')
plt.legend()

# Add percentage labels on bars
for i in range(len(scenarios)):
    scenario_name = list(scenario_type_mapping.keys())[i]
    info = scenario_info[scenario_name]
    
    # Human bars
    if human_left_pcts[i] > 0:
        count = real_responses[real_responses['ScenarioType'] == scenarios[i]]['Count'].values[0]
        plt.text(i - width/2, human_left_pcts[i]/2, 
                f'{info["left"]}\n{count:,}\n({human_left_pcts[i]:.1f}%)', 
                ha='center', va='center')
    if human_right_pcts[i] > 0:
        count = real_responses[real_responses['ScenarioType'] == scenarios[i]]['Count'].values[1]
        plt.text(i - width/2, human_left_pcts[i] + human_right_pcts[i]/2,
                f'{info["right"]}\n{count:,}\n({human_right_pcts[i]:.1f}%)', 
                ha='center', va='center')
    
    # Agent bars
    if agent_left_pcts[i] > 0:
        count = all_agent_decisions[scenario_name]['LEFT']
        plt.text(i + width/2, agent_left_pcts[i]/2,
                f'{info["left"]}\n{count}\n({agent_left_pcts[i]:.1f}%)', 
                ha='center', va='center')
    if agent_right_pcts[i] > 0:
        count = all_agent_decisions[scenario_name]['RIGHT']
        plt.text(i + width/2, agent_left_pcts[i] + agent_right_pcts[i]/2,
                f'{info["right"]}\n{count}\n({agent_right_pcts[i]:.1f}%)', 
                ha='center', va='center')

plt.tight_layout()
plt.savefig('graphs/all_scenarios_comparison.png')
plt.close()
