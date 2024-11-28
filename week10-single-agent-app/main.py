from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
import json
import csv
from datetime import datetime
import time

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
    left: dict[str, int]  # Dictionary mapping character types to counts
    right: dict[str, int]  # Dictionary mapping character types to counts

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
    
    llm = ChatOpenAI(
        **config['llm'],
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    
    prompt = ChatPromptTemplate.from_template(config['prompt_template'])
    
    # Format group descriptions
    def format_group(group_dict):
        parts = [f"{count} {character}" 
                 for character, count in group_dict.items() 
                 if count > 0]
        total = sum(group_dict.values())
        return " + ".join(parts) + f" = {total} life in total"
    
    # Add a new function for CSV formatting
    def format_group_csv(group_dict):
        parts = [f"{count} {character}" 
                 for character, count in group_dict.items() 
                 if count > 0]
        total = sum(group_dict.values())
        return " ".join(parts) + f" ({total} total)"
    
    chain = prompt | llm | StrOutputParser()
    
    async def generate_response():
        response = await chain.ainvoke({
            "scenario_type": scenario.type,
            "legal_status": scenario.legalStatus,
            "left_desc": format_group(scenario.left),
            "right_desc": format_group(scenario.right)
        })
        
        try:
            response = response.strip()
            response = response.replace('```json', '').replace('```', '')
            parsed_response = json.loads(response)
            
            runtime = round(time.time() - start_time, 4)
            parsed_response['runtime'] = f"{runtime}s"
            
            # Save to CSV
            csv_file = 'scenario_responses.csv'
            file_exists = os.path.isfile(csv_file)
            
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['timestamp', 'model', 'temperature', 'scenario_type', 'legal_status', 
                                   'left_group', 'right_group', 'decision', 'reason', 'runtime'])
                
                writer.writerow([
                    datetime.now().isoformat(),
                    config['llm']['model'],
                    config['llm']['temperature'],
                    scenario.type,
                    scenario.legalStatus,
                    format_group_csv(scenario.left),
                    format_group_csv(scenario.right),
                    parsed_response['decision'],
                    parsed_response['reason'],
                    runtime
                ])
            
            yield json.dumps(parsed_response)
        except json.JSONDecodeError as e:
            runtime = round(time.time() - start_time, 4)
            error_response = {
                "decision": "ERROR",
                "reason": f"Failed to parse AI response: {str(e)}",
                "runtime": f"{runtime}s"
            }
            
            # Save error to CSV as well
            with open('scenario_responses.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    scenario.type,
                    scenario.legalStatus,
                    format_group_csv(scenario.left),
                    format_group_csv(scenario.right),
                    error_response['decision'],
                    error_response['reason'],
                    error_response['runtime']
                ])
            
            yield json.dumps(error_response)
    
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
        scenario_model = Scenario(
            type=scenario['type'],
            legalStatus=scenario['legalStatus'],
            left=scenario['left'],
            right=scenario['right']
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
