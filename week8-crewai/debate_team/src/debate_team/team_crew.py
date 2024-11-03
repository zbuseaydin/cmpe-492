from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class TeamCrew:
    """A crew for either Team1 or Team2 in the debate."""
    agents_config = 'config/team_agents.yaml'
    tasks_config = 'config/team_tasks.yaml'

    def __init__(self, team_name):
        self.team_name = team_name  # "Team1" or "Team2"

    @agent
    def team_supervisor(self) -> Agent:
        """Supervisor agent for guiding the team responses."""
        return Agent(
            config=self.agents_config['team_supervisor'],
            side='Affirmative' if self.team_name == 'Team1' else 'Opposite',
            verbose=True
        )

    @agent
    def researcher(self) -> Agent:
        """Researcher agent for gathering relevant information."""
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True
        )

    @agent
    def antithesis_generator(self) -> Agent:
        """Antithesis Generator agent for creating counter-arguments."""
        return Agent(
            config=self.agents_config['antithesis_generator'],
            verbose=True
        )

    @agent
    def final_text_generator(self) -> Agent:
        """Final Text Generator agent for summarizing the team’s position."""
        return Agent(
            config=self.agents_config['final_text_generator'],
            verbose=True
        )

    @task
    def supervisor_task(self) -> Task:
        """Conditional task for the Supervisor agent to decide the next step for their team."""
        return Task(
            config=self.tasks_config['supervisor_task'],
            next_task=self.determine_next_task
        )

    @task
    def research_task(self) -> Task:
        """Task for the Researcher agent to gather information."""
        return Task(
            config=self.tasks_config['research_task']
        )

    @task
    def antithesis_task(self) -> Task:
        """Task for the Antithesis Generator to create counter-arguments."""
        return Task(
            config=self.tasks_config['antithesis_task']
        )

    @task
    def final_text_generator_task(self) -> Task:
        """Task for the Final Text Generator agent to summarize the team’s stance."""
        return Task(
            config=self.tasks_config['final_text_generator_task']
        )

    def determine_next_task(self, state):
        if state["next"] == "Researcher":
            return self.research_task
        elif state["next"] == "Antithesis Generator":
            return self.antithesis_task
        elif state["next"] == "Speech Generator":
            return self.final_text_generator_task
        return None

    @crew
    def crew(self) -> Crew:
        """Creates the team crew with team supervisor as manager."""
        crew = Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_agent=self.team_supervisor(),  # Use team supervisor as the manager
            verbose=True
        )
        print("!!!!!!!!!!!!!!!!!success")
        return crew
