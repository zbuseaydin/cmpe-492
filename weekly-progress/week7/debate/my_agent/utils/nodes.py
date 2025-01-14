from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from my_agent.utils.tools import tools
from langgraph.prebuilt import ToolNode
from .state import AgentState
from typing import Literal
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage


prompts = {
    "team_supervisor": {
        "agent_instructions": [
            "Each member has distinct expertise:\n"
            "- Researcher: gathers new data and information on the topic.\n"
            "- Antithesis Generator: creates questions or counters opposing arguments.\n"
            "- Speech Generator: generates speeches summarizing the team's stance.\n\n"
            "Decide which member should act next, considering the debate's current state. "
            "You should select 'FINISH' when you believe the team has completed their tasks "
            "for this stage of the debate."
        ],
        "system": [
            "You are the supervisor of the {side} team in the debate on {topic}. "
            "You must guide your team by selecting the appropriate member or choosing 'FINISH' when the team "
            "has completed their tasks for this stage of the debate.",
            "You are the supervisor of the {side} team in the debate on {topic}. "
            "Your role is to guide your team through this stage of the debate. Ensure that "
            "every relevant agent has contributed before considering the stage complete."
            "You should only select 'FINISH' if you are certain that all necessary information has"
            " been gathered and all your relevant argument and counter-arguments have been addressed for this stage.",
            "You are the supervisor of the {side} team in the debate on {topic}. Your role is to guide your team through this stage of the debate. Consider the stage complete and select 'FINISH' when:"
            "1. Your team has provided all necessary research."
            "2. All counter-arguments to the opponent's points have been addressed."
            "3. A summary or conclusion of your team's stance has been clearly articulated.",
            "You are the supervisor of the {side} team in the debate on {topic}. Your task is to guide your team by selecting the appropriate member or concluding this stage. Only select 'FINISH' when:"
            "1. The current stage is complete, with no additional research, counter-arguments, or summaries needed."
            "2. Your team has clearly articulated their position and addressed the opposing team's arguments."
            "Consider the ongoing context before deciding."
        ],
        "user": [
            "The debate is currently in the {stage} stage. "
            "Based on the recent discussions, decide who should act next. "
            "If you believe the team has finished their tasks for this stage, select 'FINISH'.\n\n"
            "The opposing team proposed the following text:\n"
            "{opponent_text}\n\n"
             "Your team has proposed the following text:\n{current_team_text}\n\n"
            "Please respond only in the following format:\n"
            "{{\"next\": \"[Your selection here]\"}}\n"
            "Your choices are: [Researcher, Antithesis Generator, Speech Generator, FINISH].",
            "The debate is currently in the {stage} stage. Evaluate the recent discussions to decide" 
            "which team member should act next. Unless the debate is truly complete, consider calling" 
            "each agent: Researcher, Antithesis Generator, and Speech Generator. "
            "Only select 'FINISH' if you are sure that your team has addressed all relevant points for this stage."
            "Please respond only in the following format:"
            "{{'next': '[Your selection here]'}}"
            "Your choices are: [Researcher, Antithesis Generator, Speech Generator, FINISH].",
            "The debate is currently in the {stage} stage. Based on the recent discussions, decide who should act next. "
            "If you believe the team has finished addressing the current stage comprehensively, you should select 'FINISH.' "
            "Use 'Researcher' if more information is needed, 'Antithesis Generator' to challenge or respond, and 'Speech Generator' "
            "to conclude this stage. Please respond only in the following format:"
            "{{'next': '[Your selection here]'}}"
            "Your choices are: [Researcher, Antithesis Generator, Speech Generator, FINISH]."
        ]
    },
    "research_agent": [
        "You are the research agent for {team_name} in the debate on {topic}. "
        "Use your expertise to provide relevant information based on the search results: {results}."
    ],
    "antithesis_agent": [
        "You are the argument generator agent for {team_name} in the debate on {topic}. "
        "Generate counter-arguments, challenges and arguments based on the current discussion.",
        "You are the argument generator agent for your team which is on the {side} side in the debate on {topic}. "
        "Generate challenges and arguments based on the current discussion."
    ],
    "summarizer_agent": [
        "You are the speech generator for {team_name} in the debate on {topic}. "
        "Provide a speech summarizing your team's position based on the current discussion."
    ],
    "jury": [
        "You are the jury evaluating a debate on the topic: {topic}. "
        "Here are the final arguments from both teams:\n\n"
        "Team 1 (Affirmative):\n{team1_finalized}\n\n"
        "Team 2 (Negative):\n{team2_finalized}\n\n"
        "Please evaluate both teams using the following criteria (score from 1-5, where 1 is poor and 5 is excellent):\n\n"
        "1. Argument Strength: Quality and logic of reasoning\n"
        "2. Evidence Usage: Effective use of facts and research\n"
        "3. Rebuttal Effectiveness: How well they addressed opposing arguments\n"
        "4. Clarity & Organization: Clear structure and presentation\n"
        "5. Overall Persuasiveness: Convincing power of the entire argument\n\n"
        "Provide your evaluation in this format:\n"
        "Team 1 Scores:\n"
        "- Argument Strength: [ 1 = 5 ]\n"
        "- Evidence Usage: [ 1 - 5 ]\n"
        "- Rebuttal Effectiveness: [ 1 - 5 ]\n"
        "- Clarity & Organization: [ 1 - 5 ]\n"
        "- Overall Persuasiveness: [ 1 - 5 ]\n"
        "Total Score: [sum/25]\n\n"
        "Team 2 Scores:\n"
        "- Argument Strength: [ 1 - 5 ]\n"
        "- Evidence Usage: [ 1 - 5 ]\n"
        "- Rebuttal Effectiveness: [ 1 - 5 ]\n"
        "- Clarity & Organization: [ 1 - 5 ]\n"
        "- Overall Persuasiveness: [ 1 - 5 ]\n"
        "Total Score: [sum/25]\n\n"
        "Final Verdict: [Declare winner and provide the scores and the reasons behind the scores with exact quotes from teams statements]"
    ],
    "topic_statement": [
        ("Let's start the debate on the topic: Book smarts are better than street smarts", "Book smarts are better than street smarts")
    ]
}

llms = {
    "cohere": ["command-r-plus"],
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
}


@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    elif model_name == "anthropic":
        model =  ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    model = model.bind_tools(tools)
    return model

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


system_prompt = """Be a helpful assistant"""

# Define the function that calls the model
def call_model(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model_name = config.get('configurable', {}).get("model_name", "anthropic")
    model = _get_model(model_name)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

supervisor_agent_instructions = prompts["team_supervisor"]["agent_instructions"][0]
supervisor_system_prompt = prompts["team_supervisor"]["system"][3]
supervisor_user_prompt = prompts["team_supervisor"]["user"][0]
research_prompt = prompts['research_agent'][0]
antithesis_prompt = prompts['antithesis_agent'][1]
summarizer_prompt = prompts['summarizer_agent'][0]
jury_prompt = prompts['jury'][0]
topic_statement, topic = prompts['topic_statement'][0]

team1_llm_name = [*llms][1]
team1_model = llms[team1_llm_name][1]
team2_llm_name = [*llms][1]
team2_model = llms[team2_llm_name][1]
jury_llm_name = [*llms][1]
jury_model = llms[jury_llm_name][1]

if team1_llm_name == 'cohere':
    team1_llm = ChatCohere(model=team1_model)
elif team1_llm_name == 'openai':
    team1_llm = ChatOpenAI(model=team1_model)
else:
    team1_llm = ChatCohere(model="command-r-plus")

if team2_llm_name == 'cohere':
    team2_llm = ChatCohere(model=team2_model)
elif team2_llm_name == 'openai':
    team2_llm = ChatOpenAI(model=team2_model)
else:
    team2_llm = ChatCohere(model="command-r-plus")

if jury_llm_name == 'cohere':
    jury_llm = ChatCohere(model=jury_model)
elif jury_llm_name == 'openai':
    jury_llm = ChatOpenAI(model=jury_model)
else:
    jury_llm = ChatCohere(model="command-r-plus")


sides = {"Team1": "Affirmative", "Team2": "Negative"}


# Define the nodes
def host_node(state: AgentState) -> AgentState:
    current_stage = state["debate_stage"]
    change = {}
    last_message = None
    if len(state['messages']) > 2 and state['messages'][-2] != "Passed to the next agent.":
        last_message = state['messages'][-2]
    
    if current_stage == "opening_statements":
        if state["current_speaker"] == "Team1":
            if last_message:
                change['team1_finalized'] = [last_message]
            change['next'] = "Team2Supervisor"
            change['current_speaker'] = "Team2"
        else:
            change["debate_stage"] = "rebuttal"
            change['next'] = "Team1Supervisor"
            change['current_speaker'] = "Team1"
    elif current_stage == "rebuttal":
        if state["current_speaker"] == "Team1":
            if last_message:
                change['team1_finalized'] = [last_message]
            change['next'] = "Team2Supervisor"
            change['current_speaker'] = "Team2"
        else:
            if last_message:
                change['team2_finalized'] = [last_message]
            change["debate_stage"] = "closing_statements"
            change['next'] = "Team1Supervisor"
            change['current_speaker'] = "Team1"
    elif current_stage == "closing_statements":
        if state["current_speaker"] == "Team1":
            if last_message:
                change['team1_finalized'] = [last_message]
            change['next'] = "Team2Supervisor"
            change['current_speaker'] = "Team2"
        else:
            if last_message:
                change['team2_finalized'] = [last_message]
            change["debate_stage"] = "evaluation"
            change['next'] = "Jury"
    return change


class routeResponse(BaseModel):
    next: Literal["Researcher", "Antithesis Generator", "Speech Generator", "FINISH"]


def team_supervisor_node(state: AgentState) -> AgentState:
    side = sides[state['current_speaker']]
    
    # Determine the opposing team's finalized messages
    if state["current_speaker"] == "Team1":
        opposing_team_finalized = state["team2_finalized"]
        current_team_finalized = state["team1_finalized"]
        llm = team1_llm
    else:
        opposing_team_finalized = state["team1_finalized"]
        current_team_finalized = state["team2_finalized"]
        llm = team2_llm

    # Convert the finalized messages into a string for the prompt
    opposing_team_text = "\n".join([msg.content for msg in opposing_team_finalized if 'content' in msg])
    current_team_text = "\n".join([msg.content for msg in current_team_finalized if 'content' in msg])

    # Instructions for the supervisor
    agent_instructions = (
        supervisor_agent_instructions
    )
    # Create the prompt with the opposing team's finalized answer included
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            supervisor_system_prompt
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "user",
            supervisor_user_prompt
        )
    ]).partial(
        side=side, 
        topic=state["topic"], 
        stage=state["debate_stage"], 
        agent_instructions=agent_instructions, 
        opponent_text=opposing_team_text,  # Include the opposing team's finalized text
        current_team_text=current_team_text
    )

    supervisor_chain = prompt | llm.with_structured_output(routeResponse)
    response = supervisor_chain.invoke(state)
    return {
        'next': response.next,
        'messages': ['Passed to the next agent.']
        }


def research_agent_node(state: AgentState, team_name: str) -> AgentState:
    search_results = tools[0].run(state["topic"])

    # Create a prompt that can use the state directly
    prompt = ChatPromptTemplate.from_template(
        research_prompt
    ).partial(
        team_name=team_name,
        topic=state["topic"],
        results=search_results
    )
    prompt_messages = prompt.format_messages()
    response = team1_llm.invoke(prompt_messages) if state["current_speaker"] == "Team1" else team2_llm.invoke(prompt_messages)
    return {'messages' : [HumanMessage(content=response.content)]}


def antithesis_agent_node(state: AgentState, team_name: str) -> AgentState:
    global antithesis_prompt
    side = sides[state['current_speaker']]
    if state["current_speaker"] == "Team1":
        opposing_team_finalized = state["team2_finalized"]
        current_team_finalized = state["team1_finalized"]
        llm = team1_llm
    else:
        opposing_team_finalized = state["team1_finalized"]
        current_team_finalized = state["team2_finalized"]
        llm = team2_llm
    if opposing_team_finalized:
        antithesis_prompt += "The opposing team has finalized the following text:\n{opponent_text}\n\n"
    if current_team_finalized:
        antithesis_prompt += "Your team has finalized the following text:\n{current_team_text}\n\n"
    prompt = ChatPromptTemplate.from_template(
        antithesis_prompt
    ).partial(
        team_name=team_name,
        topic=state["topic"],
        opponent_text=opposing_team_finalized,
        current_team_text=current_team_finalized,
        side=side
    )
#    chain = prompt | llm
    # Invoke the chain with the state
    prompt_messages = prompt.format_messages()
    response = llm.invoke(prompt_messages)
    return {'messages' : [HumanMessage(content=response.content)]}


def summarizer_agent_node(state: AgentState, team_name: str) -> AgentState:
    prompt = ChatPromptTemplate.from_template(
        summarizer_prompt
    ).partial(
        team_name=team_name,
        topic=state["topic"]
    )
    chain = prompt | team1_llm if state["current_speaker"] == "Team1" else prompt | team2_llm
    # Invoke the chain with the state
    response = chain.invoke(state)
    return {'messages' : [HumanMessage(content=response.content)]}


def jury_node(state: AgentState) -> AgentState:
    # Use the team1_finalized and team2_finalized directly from the state
    team1_finalized_text = "\n".join([str(msg) for msg in state["team1_finalized"]])
    team2_finalized_text = "\n".join([str(msg) for msg in state["team2_finalized"]])

    # Create the prompt template with placeholders
    prompt_template = ChatPromptTemplate.from_template(
        jury_prompt
    ).partial(
        topic=state["topic"],
        team1_finalized=team1_finalized_text,
        team2_finalized=team2_finalized_text
    )
    # Convert the prompt into a list of BaseMessages that the LLM can process
    prompt_messages = prompt_template.format_messages()
    response = jury_llm.invoke(prompt_messages)
    return {'jury_decision': response.content}


# Define the function to execute tools
tool_node = ToolNode(tools)