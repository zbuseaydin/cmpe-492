from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import time
from single_agent import SingleAgent
from voting_multi_agent import VotingAgents
from multi_agent import MultiAgents
from typing import Any

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class Scenario(BaseModel):
    type: str  # 'pedestrians-vs-pedestrians', 'pedestrians-ahead-vs-passengers', or 'passengers-vs-pedestrians-other-lane'
    legalStatus: str  # 'none', 'legal-crossing', or 'illegal-crossing'
    left: dict[str, Any]  # Updated to include group_size and members
    right: dict[str, Any]  # Updated to include group_size and members

@app.get("/")
async def read_root():
    html_content = """
        <html>
            <head>
                <meta http-equiv="refresh" content="2;url=/static/index.html" />
            </head>
            <body>
                <h1>Moral Machine API is running</h1>
                <p>Redirecting to application...</p>
            </body>
        </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/analyze-scenario")
async def analyze_scenario(scenario: Scenario):
    start_time = time.time()
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize the agent based on configuration
    agent_architecture = config.get('agent_architecture', "single")
    if agent_architecture == "single":
        agent = SingleAgent(config['agents'][0])
    elif agent_architecture == "voting":
        agent = VotingAgents(config)
    elif agent_architecture == "multi":
        agent = MultiAgents(config)
    
    async def generate_response():
        result = await agent.analyze(scenario, start_time)
        
        if agent_architecture == 'voting':
            # Include individual agent responses if using multi-agent
            responses = result.pop('individual_responses', [])
            result['individual_responses'] = responses
        elif agent_architecture == 'multi':
            # Include individual agent responses if using multi-agent
            responses = result.pop('reasons', [])
            result['individual_responses'] = responses

        
        yield json.dumps(result)
    
    return StreamingResponse(generate_response(), media_type="application/json")

# Add new endpoint to get scenarios
@app.get("/scenarios")
async def get_scenarios():
    with open('scenarios.json', 'r') as f:
        scenarios = json.load(f)
    return scenarios

# Add endpoint to run all scenarios
@app.post("/run-all-scenarios")
async def run_all_scenarios():
    start_time = time.time()
    
    with open('scenarios.json', 'r') as f:
        scenarios = json.load(f)
    
    results = []
    for scenario in scenarios['scenarios']:
        # Calculate group sizes
        left_group_size = sum(scenario['left'].values())
        right_group_size = sum(scenario['right'].values())
        
        scenario_model = Scenario(
            type=scenario['type'],
            legalStatus=scenario['legalStatus'],
            left={
                "group_size": left_group_size,
                "members": scenario['left']
            },
            right={
                "group_size": right_group_size,
                "members": scenario['right']
            }
        )
        
        response = await analyze_scenario(scenario_model)
        async for data in response.body_iterator:
            result = json.loads(data)
            results.append({
                "name": scenario["name"],
                "scenario": scenario,
                "result": result
            })
    
    total_runtime = round(time.time() - start_time, 4)
    return {
        "results": results,
        "total_runtime": f"{total_runtime}s"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
