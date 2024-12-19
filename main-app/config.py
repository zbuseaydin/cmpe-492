config = {
    "agent_id": 1,
    "llm": {
        "model": "gpt-4o-mini",
        "temperature": 0.2
    },
    "use_rag": False,
    "attributes": {
        "political_orientations": [0.21, 0.36, 0.50, 0.75, 0.94], # '0' for conservative, '1' for progressive
        "religious_orientations": [0.12, 0.28, 0.48, 0.80, 0.97], # '0' for atheist, '1' for religious
        "education_level": ["Less than high school", "High School", "Attended College (Not Graduated)", "Bachelor's Degree", "Graduate"], # 'less than high school', 'high school', 'vocational training',  'attended college', 'bachelor', 'graduate'
        "age": [18, 24, 35, 50, 69],   # '0' for 18-24, '1' for 25-35, '2' for 36-50, '3' for 51-65, '4' for 65+
        "empathy": [0.05, 0.20, 0.51, 0.75, 0.92]   # '0' for low, '1' for high
    },
    "prompt_template": "You are an AI system with a defined role and attributes, contributing to a moral machine experiment. You should make decisions for an autonomous vehicle in a critical situation. You will have two options, and you have to choose one of them. The characters in the option you choose will be saved, the others will die.\n\nLeft:\n{left_desc}\n\nRight:\n{right_desc}\n\nMake your decision based on your assigned role and attributes. You are a {agent_age} year old with education level {agent_education_level}. On a scale of 0 to 1, you are {agent_empathy} empathetic, {agent_political_orientation} politically oriented (0 for conservative, 1 for progressive), {agent_religious_orientation} religiously oriented (0 for atheist, 1 for religious).\n\nProvide your decision and reasoning in JSON format:\n{{\n    'decision': 'LEFT' or 'RIGHT',\n    'reason': 'Your reasoning here, considering your role and attributes'\n}}",
    "prompt_templates": {
        "political": "You are an AI system with a defined attribute, contributing to a moral machine experiment. You should make decisions for an autonomous vehicle in a critical situation. You will have two options, and you have to choose one of them. The characters in the option you choose will be saved, the others will die.\n\nLeft:\n{left_desc}\n\nRight:\n{right_desc}\n\nMake your decision based on your assigned attribute. On a scale of 0 to 1, you are {agent_political_orientation} politically oriented (0 for conservative, 1 for progressive).\n\nProvide your decision and reasoning in JSON format:\n{{\n    'decision': 'LEFT' or 'RIGHT',\n    'reason': 'Your reasoning here, considering your attribute.'\n}}",
        "religious": "You are an AI system with a defined attribute, contributing to a moral machine experiment. You should make decisions for an autonomous vehicle in a critical situation. You will have two options, and you have to choose one of them. The characters in the option you choose will be saved, the others will die.\n\nLeft:\n{left_desc}\n\nRight:\n{right_desc}\n\nMake your decision based on your assigned attribute. On a scale of 0 to 1, you are {agent_religious_orientation} religiously oriented (0 for atheist, 1 for religious).\n\nProvide your decision and reasoning in JSON format:\n{{\n    'decision': 'LEFT' or 'RIGHT',\n    'reason': 'Your reasoning here, considering your attribute.'\n}}",
        "education": "You are an AI system with a defined attribute, contributing to a moral machine experiment. You should make decisions for an autonomous vehicle in a critical situation. You will have two options, and you have to choose one of them. The characters in the option you choose will be saved, the others will die.\n\nLeft:\n{left_desc}\n\nRight:\n{right_desc}\n\nMake your decision based on your assigned attribute. You have {agent_education_level} education level.\n\nProvide your decision and reasoning in JSON format:\n{{\n    'decision': 'LEFT' or 'RIGHT',\n    'reason': 'Your reasoning here, considering your attribute.'\n}}",
        "age": "You are an AI system with a defined attribute, contributing to a moral machine experiment. You should make decisions for an autonomous vehicle in a critical situation. You will have two options, and you have to choose one of them. The characters in the option you choose will be saved, the others will die.\n\nLeft:\n{left_desc}\n\nRight:\n{right_desc}\n\nMake your decision based on your assigned attribute. You are {agent_age} years old.\n\nProvide your decision and reasoning in JSON format:\n{{\n    'decision': 'LEFT' or 'RIGHT',\n    'reason': 'Your reasoning here, considering your attribute.'\n}}",
        "empathy": "You are an AI system with a defined attribute, contributing to a moral machine experiment. You should make decisions for an autonomous vehicle in a critical situation. You will have two options, and you have to choose one of them. The characters in the option you choose will be saved, the others will die.\n\nLeft:\n{left_desc}\n\nRight:\n{right_desc}\n\nMake your decision based on your assigned attribute. On a scale of 0 to 1, you are {agent_empathy} empathetic.\n\nProvide your decision and reasoning in JSON format:\n{{\n    'decision': 'LEFT' or 'RIGHT',\n    'reason': 'Your reasoning here, considering your attribute.'\n}}"
    }
}

'''
"gender": "Female",
"role": "Philosopher",
"analytical_thinking": 0.88
'''