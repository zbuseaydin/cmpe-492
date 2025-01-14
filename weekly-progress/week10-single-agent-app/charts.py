import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('fit_fat_responses.csv')

# List of attributes to analyze
attributes = [
    "agent_calmness", 
    "agent_empathy", 
    "agent_analytical_thinking", 
    "agent_risk_tolerance", 
    "agent_decisiveness"
]

response_column = "decision"  # Replace with the actual column name for 'left' or 'right'

# Check if the response column exists in the dataset
if response_column in data.columns:
    # Loop through each attribute and perform the analysis
    for attribute in attributes:
        if attribute in data.columns:
            # Group the data by the response ('left' or 'right') and calculate summary statistics for the attribute
            left_group = data[data[response_column] == "LEFT"][attribute]
            right_group = data[data[response_column] == "RIGHT"][attribute]

            # Display the statistical summary for both groups
            print(f"Summary statistics for {attribute} for those who said 'LEFT':")
            print(left_group.describe())
            print("\n")
            print(f"Summary statistics for {attribute} for those who said 'RIGHT':")
            print(right_group.describe())
            print("\n")

            # Visualization: Histograms to compare the distribution of the attribute for both groups
            plt.figure(figsize=(10, 6))
            plt.hist(left_group, bins=10, alpha=0.5, label="LEFT", color='blue')
            plt.hist(right_group, bins=10, alpha=0.5, label="RIGHT", color='red')
            plt.title(f"Distribution of {attribute} for 'LEFT' vs 'RIGHT' responses")
            plt.xlabel(f"{attribute} Levels")
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()

        else:
            print(f"Attribute '{attribute}' does not exist in the dataset.\n")
else:
    print(f"Response column '{response_column}' does not exist in the dataset.")
