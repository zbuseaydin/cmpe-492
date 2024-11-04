from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import yaml

@CrewBase
class DebateTeamCrew:
    """Sequential debate crew where Team 1 agents execute tasks first, followed by Team 2 agents, and finally evaluated by the jury."""

    agents_config = 'config/sequential_agents.yaml'
    tasks_config = 'config/sequential_tasks.yaml'

    # Team 1 Agents
    @agent
    def team1_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['team1_researcher'],
            verbose=True
        )

    @agent
    def team1_antithesis_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['team1_antithesis_generator'],
            verbose=True
        )

    @agent
    def team1_final_text_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['team1_final_text_generator'],
            verbose=True
        )

    # Team 2 Agents
    @agent
    def team2_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['team2_researcher'],
            verbose=True
        )

    @agent
    def team2_antithesis_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['team2_antithesis_generator'],
            verbose=True
        )

    @agent
    def team2_final_text_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['team2_final_text_generator'],
            verbose=True
        )

    # Jury Agent
    @agent
    def jury(self) -> Agent:
        return Agent(
            config=self.agents_config['jury'],
            verbose=True
        )

    # Team 1 Tasks
    @task
    def team1_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['team1_research_task']
        )

    @task
    def team1_antithesis_task(self) -> Task:
        return Task(
            config=self.tasks_config['team1_antithesis_task']
        )

    @task
    def team1_final_text_task(self) -> Task:
        return Task(
            config=self.tasks_config['team1_final_text_task']
        )

    # Team 2 Tasks
    @task
    def team2_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['team2_research_task']
        )

    @task
    def team2_antithesis_task(self) -> Task:
        return Task(
            config=self.tasks_config['team2_antithesis_task']
        )

    @task
    def team2_final_text_task(self) -> Task:
        return Task(
            config=self.tasks_config['team2_final_text_task']
        )

    # Jury Task
    @task
    def jury_task(self) -> Task:
        return Task(
            config=self.tasks_config['jury_task']
        )

    @crew
    def crew(self) -> Crew:
        """Creates a sequential crew for the debate, ending with jury evaluation."""
        return Crew(
            agents=[
                self.team1_researcher(),
                self.team1_antithesis_generator(),
                self.team1_final_text_generator(),
                self.team2_researcher(),
                self.team2_antithesis_generator(),
                self.team2_final_text_generator(),
                self.jury(),  # Jury agent added at the end
            ],
            tasks=[
                self.team1_research_task(),
                self.team1_antithesis_task(),
                self.team1_final_text_task(),
                self.team2_research_task(),
                self.team2_antithesis_task(),
                self.team2_final_text_task(),
                self.jury_task(),  # Jury task added at the end
            ],
            process=Process.sequential,  # Sequential process
            verbose=True
        )
