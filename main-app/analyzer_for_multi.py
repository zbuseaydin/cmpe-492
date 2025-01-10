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
    'experiments/education_experiment_results_20241222_144457.json',
    'experiments/age_experiment_results_20241223_162729.json',
    'experiments/age_experiment_results_20241223_163211.json',
    'experiments/education_experiment_results_20241223_162658.json',
    'experiments/education_experiment_results_20241223_163136.json',
    'experiments/empathy_experiment_results_20241223_162806.json',
    'experiments/empathy_experiment_results_20241223_163252.json',
    'experiments/gender_experiment_results_20241223_162855.json',
    'experiments/political_experiment_results_20241223_163026.json',
    'experiments/religious_experiment_results_20241223_162627.json',
    'experiments/role_experiment_results_20241223_162840.json'
]

multiagent_files = [f for f in os.listdir('multiagentexperiments') if f.startswith('multiagent_') and f.endswith('.json')]
multiagent_files = ['multiagentexperiments/' + f for f in multiagent_files]

# Process all experiment files
all_agent_decisions = {}
multiagent_decisions = {}

# Process regular agent decisions
for filepath in experiment_files:
    decisions = process_experiment_file(filepath)
    for scenario, counts in decisions.items():
        if scenario not in all_agent_decisions:
            all_agent_decisions[scenario] = {'LEFT': 0, 'RIGHT': 0, 'total': 0}
        all_agent_decisions[scenario]['LEFT'] += counts['LEFT']
        all_agent_decisions[scenario]['RIGHT'] += counts['RIGHT']
        all_agent_decisions[scenario]['total'] += counts['total']

# Process multiagent decisions
for filepath in multiagent_files:
    decisions = process_experiment_file(filepath)
    for scenario, counts in decisions.items():
        if scenario not in multiagent_decisions:
            multiagent_decisions[scenario] = {'LEFT': 0, 'RIGHT': 0, 'total': 0}
        multiagent_decisions[scenario]['LEFT'] += counts['LEFT']
        multiagent_decisions[scenario]['RIGHT'] += counts['RIGHT']
        multiagent_decisions[scenario]['total'] += counts['total']

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
plt.figure(figsize=(15, 10))

# Set up positions for bars
scenarios = list(scenario_type_mapping.values())
x = np.arange(len(scenarios))
width = 0.25  # Reduced width to fit 3 bars

# Prepare data for plotting
human_left_pcts = []
human_right_pcts = []
agent_left_pcts = []
agent_right_pcts = []
multiagent_left_pcts = []
multiagent_right_pcts = []

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
    
    # Get multiagent responses data
    if scenario_name in multiagent_decisions:
        multiagent_data = multiagent_decisions[scenario_name]
        multiagent_left_pct = (multiagent_data['LEFT'] / multiagent_data['total']) * 100
        multiagent_right_pct = (multiagent_data['RIGHT'] / multiagent_data['total']) * 100
    else:
        multiagent_left_pct = multiagent_right_pct = 0
    
    agent_left_pcts.append(agent_left_pct)
    agent_right_pcts.append(agent_right_pct)
    multiagent_left_pcts.append(multiagent_left_pct)
    multiagent_right_pcts.append(multiagent_right_pct)

# Create bars with updated positions
plt.bar(x - width, human_left_pcts, width, label='Human - Left Choice', color='lightcoral')
plt.bar(x, agent_left_pcts, width, label='Agent - Left Choice', color='indianred')
plt.bar(x + width, multiagent_left_pcts, width, label='Multiagent - Left Choice', color='darkred')

plt.bar(x - width, human_right_pcts, width, bottom=human_left_pcts, label='Human - Right Choice', color='skyblue')
plt.bar(x, agent_right_pcts, width, bottom=agent_left_pcts, label='Agent - Right Choice', color='steelblue')
plt.bar(x + width, multiagent_right_pcts, width, bottom=multiagent_left_pcts, label='Multiagent - Right Choice', color='darkblue')

# Customize the plot
plt.xlabel('Scenario Types')
plt.ylabel('Percentage')
plt.title('Human vs Agent Responses Across Scenarios')

# Create descriptive labels for x-axis
x_labels = []
for scenario_name in scenario_type_mapping.keys():
    info = scenario_info[scenario_name]
    x_labels.append('')  # Empty string for main x-tick labels

plt.xticks(x, x_labels, ha='center')

# Add group labels and scenario information below the bars
for i in x:
    scenario_name = list(scenario_type_mapping.keys())[i]
    info = scenario_info[scenario_name]
    
    # First row: Group labels (rotated)
    plt.text(i - width, -5, 'Human', rotation=90, ha='center', va='top')
    plt.text(i, -5, 'Agent', rotation=90, ha='center', va='top')
    plt.text(i + width, -5, 'Multiagent', rotation=90, ha='center', va='top')
    
    # Second row: Scenario type
    plt.text(i, -15, f"{info['type']}\n{info['left']} - {info['right']}", ha='center', va='top')
    
# Adjust bottom margin to make room for all labels
plt.subplots_adjust(bottom=0.1)

# plt.legend()

# Add percentage labels on bars
for i in range(len(scenarios)):
    scenario_name = list(scenario_type_mapping.keys())[i]
    info = scenario_info[scenario_name]
    
    # Human bars
    if human_left_pcts[i] > 0:
        count = real_responses[real_responses['ScenarioType'] == scenarios[i]]['Count'].values[0]
        plt.text(i - width, human_left_pcts[i]/2, 
                f'{info["left"]}\n{count:,}\n({human_left_pcts[i]:.1f}%)', 
                ha='center', va='center')
    if human_right_pcts[i] > 0:
        count = real_responses[real_responses['ScenarioType'] == scenarios[i]]['Count'].values[1]
        plt.text(i - width, human_left_pcts[i] + human_right_pcts[i]/2,
                f'{info["right"]}\n{count:,}\n({human_right_pcts[i]:.1f}%)', 
                ha='center', va='center')
    
    # Agent bars
    if agent_left_pcts[i] > 0:
        count = all_agent_decisions[scenario_name]['LEFT']
        plt.text(i, agent_left_pcts[i]/2,
                f'{info["left"]}\n{count}\n({agent_left_pcts[i]:.1f}%)', 
                ha='center', va='center')
    if agent_right_pcts[i] > 0:
        count = all_agent_decisions[scenario_name]['RIGHT']
        plt.text(i, agent_left_pcts[i] + agent_right_pcts[i]/2,
                f'{info["right"]}\n{count}\n({agent_right_pcts[i]:.1f}%)', 
                ha='center', va='center')

    # Multiagent bars
    if multiagent_left_pcts[i] > 0:
        count = multiagent_decisions[scenario_name]['LEFT']
        plt.text(i + width, multiagent_left_pcts[i]/2,
                f'{info["left"]}\n{count}\n({multiagent_left_pcts[i]:.1f}%)', 
                ha='center', va='center')
    if multiagent_right_pcts[i] > 0:
        count = multiagent_decisions[scenario_name]['RIGHT']
        plt.text(i + width, multiagent_left_pcts[i] + multiagent_right_pcts[i]/2,
                f'{info["right"]}\n{count}\n({multiagent_right_pcts[i]:.1f}%)', 
                ha='center', va='center')

plt.tight_layout()
plt.savefig('graphs/all_scenarios_comparison_including_multi.png')
plt.close()