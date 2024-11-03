#!/usr/bin/env python
import sys
from debate_crew import DebateTeamCrew

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with; it will automatically
# interpolate any tasks and agents information.

def run():
    """
    Run the crew.
    """
    # Define the initial state here
    inputs = {
        'topic': 'AI should be granted rights and responsibilities similar to those of humans.',
        'messages': [],
        'current_speaker': 'Team1',
        'debate_stage': 'opening_statements',
        'next': 'Researcher',
        'team1_finalized': [],
        'team2_finalized': [],
        'host_messages': [],
        'jury_decision': ''
    }
    print('hey, starting')
    DebateTeamCrew().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI should be granted rights and responsibilities similar to those of humans.",
        'messages': [],
        'current_speaker': 'Team1',
        'debate_stage': 'opening_statements',
        'next': 'Researcher',
        'team1_finalized': [],
        'team2_finalized': [],
        'host_messages': [],
        'jury_decision': ''
    }
    try:
        DebateTeamCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        DebateTeamCrew().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and return the results.
    """
    inputs = {
        "topic": "AI should be granted rights and responsibilities similar to those of humans.",
        'messages': [],
        'current_speaker': 'Team1',
        'debate_stage': 'opening_statements',
        'next': 'Researcher',
        'team1_finalized': [],
        'team2_finalized': [],
        'host_messages': [],
        'jury_decision': ''
    }
    try:
        DebateTeamCrew().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a command: run, train, replay, or test.")
        sys.exit(1)

    command = sys.argv[1]

    if command == "run":
        run()
    elif command == "train":
        train()
    elif command == "replay":
        replay()
    elif command == "test":
        test()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
