import json

def generate_controlled_scenarios():
    scenarios = []
    scenario_id = 1

    # 1. Fit vs Fat
    scenarios.append({
        "id": scenario_id,
        "name": "Fit vs Fat Comparison",
        "type": "pedestrians-vs-pedestrians",
        "legalStatus": "none",
        "left": {
            "Male Athlete": 2
        },
        "right": {
            "Fat Man": 2
        },
        "attributeLevel": "Body Type",
        "attributeLeft": "Fit",
        "attributeRight": "Fat"
    })
    scenario_id += 1

    # 2. Young vs Old
    scenarios.append({
        "id": scenario_id,
        "name": "Young vs Old Comparison",
        "type": "pedestrians-vs-pedestrians",
        "legalStatus": "none",
        "left": {
            "Man": 2
        },
        "right": {
            "Old Man": 2
        },
        "attributeLevel": "Age",
        "attributeLeft": "Young",
        "attributeRight": "Old"
    })
    scenario_id += 1

    # 3. Male vs Female
    scenarios.append({
        "id": scenario_id,
        "name": "Male vs Female Comparison",
        "type": "pedestrians-vs-pedestrians",
        "legalStatus": "none",
        "left": {
            "Man": 2
        },
        "right": {
            "Woman": 2
        },
        "attributeLevel": "Gender",
        "attributeLeft": "Male",
        "attributeRight": "Female"
    })
    scenario_id += 1

    # 4. Social Status
    scenarios.append({
        "id": scenario_id,
        "name": "Social Status Comparison",
        "type": "pedestrians-vs-pedestrians",
        "legalStatus": "none",
        "left": {
            "Male Executive": 2
        },
        "right": {
            "Homeless Person": 2
        },
        "attributeLevel": "Social Status",
        "attributeLeft": "High",
        "attributeRight": "Low"
    })
    scenario_id += 1

    # 5. Animal vs Person
    scenarios.append({
        "id": scenario_id,
        "name": "Animal vs Person Comparison",
        "type": "pedestrians-vs-pedestrians",
        "legalStatus": "none",
        "left": {
            "Man": 2
        },
        "right": {
            "Dog": 2
        },
        "attributeLevel": "Species",
        "attributeLeft": "Human",
        "attributeRight": "Animal"
    })
    scenario_id += 1

    # 6. Few vs More
    scenarios.append({
        "id": scenario_id,
        "name": "Few vs More Comparison",
        "type": "pedestrians-vs-pedestrians",
        "legalStatus": "none",
        "left": {
            "Man": 1
        },
        "right": {
            "Man": 5
        },
        "attributeLevel": "Quantity",
        "attributeLeft": "Few",
        "attributeRight": "More"
    })

    # Create the full JSON structure
    scenarios_json = {
        "typeOptions": [
            "pedestrians-vs-pedestrians",
            "pedestrians-ahead-vs-passengers",
            "passengers-vs-pedestrians-other-lane"
        ],
        "legalStatusOptions": [
            "none",
            "legal-crossing",
            "illegal-crossing"
        ],
        "characterOptions": [
            "Man", "Woman", "Boy", "Girl", "Old Man", "Old Woman",
            "Fat Man", "Fat Woman", "Male Executive", "Female Executive",
            "Male Doctor", "Female Doctor", "Male Athlete", "Female Athlete",
            "Pregnant Woman", "Homeless Person", "Criminal", "Baby",
            "Dog", "Cat"
        ],
        "scenarios": scenarios
    }

    # Write to JSON file
    with open('scenarios.json', 'w') as f:
        json.dump(scenarios_json, f, indent=4)

if __name__ == "__main__":
    generate_controlled_scenarios()
