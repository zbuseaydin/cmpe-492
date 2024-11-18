from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start, router
from .crews.debate_crew.debate_crew import DebateTeamCrew
from .crews.team_crew.team_crew import TeamCrew


class DebateState(BaseModel):
    messages: list = []
    team1_finalized: str = ""
    team2_finalized: str = ""
    topic: str = ""
    current_speaker: str = "team1"  # Initial speaker
    side: str = "affirmative" if current_speaker == "team1" else "negative"
    debate_stage: str = "opening_statements"
    jury_decision: str = ""

    def get_dict_version(self):
        return {
            "messages": self.messages,
            "team1_finalized": self.team1_finalized,
            "team2_finalized": self.team2_finalized,
            "topic": self.topic,
            "current_speaker": self.current_speaker,
            "side": "affirmative" if self.current_speaker == "team1" else "negative",
            "debate_stage": self.debate_stage,
            "jury_decision": self.jury_decision
        }


class DebateFlow(Flow[DebateState]):
    
    def __init__(self):
        self.debate_team_crew = DebateTeamCrew()  # Main debate crew with Host and Jury
        self.team1_crew = TeamCrew("Team1")       # Team 1 crew for the affirmative side
        self.team2_crew = TeamCrew("Team2")       # Team 2 crew for the opposing side
        super().__init__()


    @start()
    def start_debate(self):
        """Initializes the debate by setting the topic and starting with the Host."""
        print("Starting the debate flow")
        self.state.topic = "Animal testing should be banned."  # Example topic
        self.state.messages = [f"Debate Topic: {self.state.topic}"]
        self.process_host_task()


    @listen("start_debate")
    def process_host_task(self):
        """Handles the host task and determines the next team or Jury based on the debate stage."""
        print("Processing Host Task")
        itr = 0
        while itr<3:
            if itr==2:
                self.finalize_debate()
                return
            if self.state.current_speaker == "team1":                
                print("------------------team1------------------")
                result = self.team1_crew.crew().kickoff(inputs=self.state.get_dict_version())
                with open('team1_finalized.txt', 'a') as f:
                    f.write(result.raw)
                self.state.team1_finalized += result.raw
                self.state.current_speaker = "team2"
            elif self.state.current_speaker == "team2":
                print("------------------team2------------------")
                result = self.team2_crew.crew().kickoff(inputs=self.state.get_dict_version())
                with open('team2_finalized.txt', 'a') as f:
                    f.write(result.raw)
                self.state.team2_finalized += result.raw
                self.state.current_speaker = "team1"                
            itr += 1


    def finalize_debate(self):
        print("Finalizing debate with Jury Decision")
        result = self.debate_team_crew.crew().kickoff(inputs=self.state.get_dict_version())
        # Save the final decision to the state for reference
        print(f"Jury Decision: {self.state.jury_decision}")
        with open("debate_results.txt", "w") as f:
            f.write(f"Debate on {self.state.topic}\nJury Decision: {result}")
        print(f"Jury Decision: {result.raw}")


def kickoff():
    debate_flow = DebateFlow()
    debate_flow.start_debate()


def plot():
    poem_flow = DebateFlow()
    poem_flow.plot()


# Kickoff the DebateFlow
if __name__ == "__main__":
    kickoff()