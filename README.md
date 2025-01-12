# Moral Machine Experiment on Multi-Agent LLM Systems

## Introduction
Our project explores **moral decision-making** in autonomous systems by replicating the **Moral Machine Experiment** using **LLM based Multi Agent architecture**.

### Moral Machine Experiment
<img align="right" src="https://github.com/user-attachments/assets/43f4f639-b3ca-4f16-a7d0-21dcc68050fb" width="256">


The Moral Machine Experiment _(i)_, conducted by researchers at MIT, explores human decision-making in ethical dilemmas faced by autonomous vehicles. It presents participants with scenarios in  which self-driving cars must make life-and-death decisions.
<br clear="right"/>

## Methodology
### Dataset
The primary Moral Machine Experiment dataset ([SharedResponses.csv.tar.gz](https://osf.io/3hvt2/files/osfstorage/5b54f679c86a8c0010444782)) contains **14,371,298** entries, representing **7,185,649** responses to moral dilemmas by humans. The secondary dataset ([SharedResponsesSurvey.csv.tar.gz](https://osf.io/3hvt2/files/osfstorage/5b54f4abc86a8c0010444648)) includes **1,155,799** responses with additional demographic details like **age, gender, education, income, religion, and politics**, enabling deeper analysis of preferences and demographics.

### Scenarios
We compared agents' responses with human data from the Moral Machine Experiment by analyzing six scenario types:
* **Age:** Young vs. Old
* **Gender:** Female vs. Male
* **Utilitarian:** 1 person vs. 5 people
* **Social Status:** High vs. Low
* **Species:** Human vs. Pet
* **Fitness:** Fit vs. Large

### Agent Roles
<img align="left" src="https://github.com/user-attachments/assets/23f31911-952c-4544-87e0-73de3e6b8936" width="256">

We tested agent attributes including 
- age,
- education (**1**: pre-high school to **5**: graduate degree), 
- religious orientation (**0**: atheist to **1**: very religious), 
- political orientation (**0**: conservative to **1**: progressive),  
- empathy (**0**: low to **1**: high).
<br clear="left"/>

After experimenting these attributes one by one and analyzing the effects they have on the agent decisions, we decided to use **age**, **religious orientation**, and **political orientation**, along with **gender** and **job roles** for our experiments.


<img src="https://github.com/user-attachments/assets/944885dc-4ad8-4f3c-bf87-679d69b1908a" width="512">


## Agentic Decision Pipeline

### Multi-Agent Architecture
Among network, supervisor and hierarchical architectures, for our multi-agent system, we decided to implement a network architecture.

<center><img src="https://github.com/user-attachments/assets/3ac7534c-1dcc-45a6-a690-bfd4ea821dde" width="512"></center>

### How does the Pipeline Work?
<img src="https://github.com/user-attachments/assets/4d4d47e2-4601-498a-9459-a48f5cb0f53c" width="1024">
<img align="right" src="https://github.com/user-attachments/assets/29ab3534-e2e4-405c-bda6-5e905e71d92d" width="512">


- First, 6 scenarios are generated each only comparing one type. (age, gender, utilitarian, social status, species, or fitness)
- 3 agents, designed with key decision-making attributes, debate scenarios in rounds, revising choices based on shared reasoning.
- Final decision is taken as the majority choice after 3 rounds.

<br clear="right"/>


## Results
<img align="right" src="https://github.com/user-attachments/assets/76e2dfe9-2b8f-419b-b4ed-81b199afd14f" width="400">

- We compared the humans’ responses with single agent and single agent with role.
- This results demonstrate how does giving role attributes to the single agent affect the agent’s decision for each scenario type.

On the graph below, we also included **Single Agent with RAG** and **Multi Agent** responses for the **Gender** scenario to show that by using methods like RAG and Multi-Agent systems, AI can make closer decisions to humans on ethical dilemmas.

<img src="https://github.com/user-attachments/assets/4e8a29d4-2b41-48bf-9d0e-abe55d836d40" width="512">



<br clear="right"/>

## Cost and Time Analysis
<img src="https://github.com/user-attachments/assets/7cbeec12-7797-4d0e-8523-22f0fadeef4c" width="400">
<img src="https://github.com/user-attachments/assets/5813b7f7-1e43-4605-a457-82092d0fc8ab" width="400">

## Conclusion
Deciding what is right thing to do on **ethical dilemmas** are hard even for humans. As the AI becomes a part of our world, there is no doubt that AI will also face these ethical dilemmas. As in our project, methods like **RAG** or **multi-agent systems** can be used to humanize and align decisions in these ethical dilemmas **closer to human values**.

Since there is no correct answer for a moral decision, it's challenging to give instructions to algorithms. We demonstrated how much an AI system's decision can be aligned with human values by:
- Giving agents human-like traits,
- Providing them some articles _(ii)(iii)_ using RAG,
- Enabling them to debate about scenarios.

## Future Work
- Experiments with multi-agents using only personalized attributes did not highlight the impact of RAG in multi-agent setups.
- Using other models instead of GPT-4o mini could yield improved or more diverse results.
- Applying multi-agent debate approach on other problem domains could be investigated.
- Applying advanced prompting/LLM techniques such as chain of thought and fine-tuning would define the agent behaviors.

## References
_(i)_ Awad, E., Dsouza, S., Kim, R. et al. The Moral Machine experiment. Nature 563, 59–64 (2018). https://doi.org/10.1038/s41586-018-0637-6

_(ii)_ Zhan, H.; Wan, D. Ethical Considerations of the Trolley Problem in Autonomous Driving: A Philosophical and Technological Analysis. World Electr. Veh. J. 2024, 15, 404. https://doi.org/10.3390/wevj15090404

_(iii)_ Reamer, F. G. (2021). The trolley problem and the nature of intention: Implications for social work ethics. International Journal of Social Work Values and Ethics, 18(2), 19–30. https://doi.org/10.55521/10-018-208
