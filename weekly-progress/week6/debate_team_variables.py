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
        "Based on these arguments, please provide a verdict on which team presented a stronger case."
    ],
    "topic_statement": [
        ("Let's start the debate on the topic: AI's impact on job markets", "AI's impact on job markets")
    ]
}

llms = {
    "cohere": ["command-r-plus"],
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
}