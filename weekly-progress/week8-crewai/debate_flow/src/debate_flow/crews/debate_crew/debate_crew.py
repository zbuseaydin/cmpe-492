from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class DebateTeamCrew:
    """Main debate crew coordinating Team1 and Team2 with the host manager."""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

#    @agent
#    def host(self) -> Agent:
#        """Host agent responsible for managing the overall debate."""
#        return Agent(
#            config=self.agents_config['host'],
#            verbose=True
#        )

    @agent
    def jury(self) -> Agent:
        """Jury agent responsible for evaluating and scoring the debate."""
        return Agent(
            config=self.agents_config['jury'],
            verbose=True
        )

#    @task
#    def host_task(self) -> Task:
#        """Task for the Host agent to manage the flow and coordinate teams."""
#        return Task(
#            config=self.tasks_config['host_task'],
#            agent=self.host(),
#            next_task=self.determine_next_task  # Decide which team crew to activate
#        )

    @task
    def jury_task(self) -> Task:
        """Task for the Jury agent to evaluate the debate and provide a final decision."""
        return Task(
            config=self.tasks_config['jury_task'],
            agent=self.jury()
        )
    

    @crew
    def crew(self) -> Crew:
        """Creates the main debate crew with the host as manager."""
        return Crew(
            agents=self.agents,  # Automatically includes all agents defined with @agent decorator
            tasks=self.tasks,  # Only include host and jury tasks
            verbose=True
        )