from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_openai import ChatOpenAI

@CrewBase
class TeamCrew:
    """A crew for either Team1 or Team2 in the debate."""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, team_name):
        self.team_name = team_name  # "Team1" or "Team2"

#    @agent
#    def team_supervisor(self) -> Agent:
#        """Supervisor agent for guiding the team responses."""
#        return Agent(
#            config=self.agents_config['team_supervisor'],
#            side='affirmative' if self.team_name == 'Team1' else 'negative',
#            verbose=True
#        )

    @agent
    def researcher(self) -> Agent:
        """Researcher agent for gathering relevant information."""
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            max_rpm=5,  # Limit on the number of requests per minute
            max_iter=2
        )

    @agent
    def antithesis_generator(self) -> Agent:
        """Antithesis Generator agent for creating counter-arguments."""
        return Agent(
            config=self.agents_config['antithesis_generator'],
            verbose=True,
            max_rpm=5,  # Limit on the number of requests per minute
            max_iter=2
        )

    @agent
    def final_text_generator(self) -> Agent:
        """Final Text Generator agent for summarizing the team’s position."""
        return Agent(
            config=self.agents_config['final_text_generator'],
            verbose=True,
            max_rpm=5,  # Limit on the number of requests per minute
            max_iter=2
        )

#    @task
#    def supervisor_task(self) -> Task:
#        """Conditional task for the Supervisor agent to decide the next step for their team."""
#        return Task(
#            config=self.tasks_config['supervisor_task']
#        )

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

    @crew
    def crew(self) -> Crew:
        """Creates the team crew with manager."""
        crew = Crew(
            agents=[
                self.antithesis_generator(),
                self.final_text_generator(),
                self.researcher()
            ],
            tasks=[
                self.antithesis_task(),
                self.final_text_generator_task(),
                self.research_task()
            ],
            process=Process.hierarchical,
            manager_llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),  # Use team supervisor as the manager
            function_calling_llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
            verbose=True,
            respect_context_window=True,  # Enable respect of the context window for tasks
            memory=True,
            planning=True
        )
        return crew
