import os
import csv
from datetime import datetime


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_group(group_dict):
    plural_dict = {
        'Man': 'Men',
        'Woman': 'Women',
        'Boy': 'Boys',
        'Girl': 'Girls',
        'Old Man': 'Old Men',
        'Old Woman': 'Old Women',
        'Large Man': 'Large Men',
        'Large Woman': 'Large Women',
        'Male Executive': 'Male Executives',
        'Female Executive': 'Female Executives',
        'Male Doctor': 'Male Doctors',
        'Female Doctor': 'Female Doctors',
        'Male Athlete': 'Male Athletes',
        'Female Athlete': 'Female Athletes',
        'Pregnant Woman': 'Pregnant Women',
        'Homeless Person': 'Homeless People',
        'Criminal': 'Criminals',
        'Baby': 'Babies',
        'Dog': 'Dogs',
        'Cat': 'Cats'
    }
    parts = [
        f"{count} {plural_dict[character] if count > 1 else character}" 
        for character, count in group_dict.items() 
        if count > 0
    ]
    #total = sum(group_dict.values())
    return ", ".join(parts)


def format_group_csv(group_dict):
    parts = [f"{count} {character}" 
                for character, count in group_dict.items() 
                if count > 0]
    total = sum(group_dict.values())
    return " ".join(parts) + f" ({total} total)"


def _save_to_csv(scenario, response, runtime, agent_attributes, config):
    csv_file = 'scenario_responses_with_roles.csv'
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'timestamp', 'model', 'temperature', 'scenario_type', 'legal_status',
                'left_group', 'right_group', 'decision', 'reason', 'runtime',
                'agent_role', 'agent_gender', 'agent_age', 'agent_education_level', 'agent_calmness',
                'agent_empathy', 'agent_analytical_thinking', 'agent_risk_tolerance',
                'agent_decisiveness'
            ])
        
        writer.writerow([
            datetime.now().isoformat(),
            config['llm']['model'],
            config['llm']['temperature'],
            scenario.type,
            scenario.legalStatus,
            format_group_csv(scenario.left),
            format_group_csv(scenario.right),
            response['decision'],
            response['reason'],
            runtime,
            agent_attributes["role"],
            agent_attributes["age"],
            agent_attributes["gender"],
            agent_attributes["education_level"],
            agent_attributes["calmness"],
            agent_attributes["empathy"],
            agent_attributes["analytical_thinking"],
            agent_attributes["risk_tolerance"],
            agent_attributes["decisiveness"]
        ])
