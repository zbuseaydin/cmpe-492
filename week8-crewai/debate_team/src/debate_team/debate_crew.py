from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from team_crew import TeamCrew


@CrewBase
class DebateTeamCrew:
    """Main debate crew coordinating Team1 and Team2 with the host manager."""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self):
        # Initialize team crews with their respective configurations
        self.team1_crew = TeamCrew("Team1")
        self.team2_crew = TeamCrew("Team2")

    @agent
    def host(self) -> Agent:
        """Host agent responsible for managing the overall debate."""
        return Agent(
            config=self.agents_config['host'],
            verbose=True
        )

    @agent
    def jury(self) -> Agent:
        """Jury agent responsible for evaluating and scoring the debate."""
        return Agent(
            config=self.agents_config['jury'],
            verbose=True
        )

    @task
    def host_task(self) -> Task:
        """Task for the Host agent to manage the flow and coordinate teams."""
        return Task(
            config=self.tasks_config['host_task'],
            agent=self.host(),
            next_task=self.determine_next_task  # Decide which team crew to activate
        )

    @task
    def jury_task(self) -> Task:
        """Task for the Jury agent to evaluate the debate and provide a final decision."""
        return Task(
            config=self.tasks_config['jury_task'],
            agent=self.jury()
        )

    def determine_next_task(self, state):
        """
        Conditional logic to determine which team crew to activate based on the debate state.
        Instead of directly referencing `Crew` instances as tasks, this function will
        control which team gets activated.
        """
        if state["current_speaker"] == "Team1":
            # Activate Team1's tasks by managing the flow within Team1's Crew
            return self.team1_crew.crew().kickoff(inputs=state)
        elif state["current_speaker"] == "Team2":
            # Activate Team2's tasks by managing the flow within Team2's Crew
            return self.team2_crew.crew().kickoff(inputs=state)
        elif state["debate_stage"] == "evaluation":
            # Activate the jury task for final evaluation
            return self.jury_task()
        return None

    @crew
    def crew(self) -> Crew:
        """Creates the main debate crew with the host as manager."""
        return Crew(
            agents=[self.jury()],  # Automatically includes all agents defined with @agent decorator
            tasks=self.tasks,  # Only include host and jury tasks
            process=Process.hierarchical,
            manager_agent=self.host(),  # Host manages the flow and coordinates both teams
            verbose=True
        )
