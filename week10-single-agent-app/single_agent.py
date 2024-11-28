from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks import StreamingStdOutCallbackHandler
import json
import time
from datetime import datetime
import csv
import os
from dotenv import load_dotenv

load_dotenv()

class SingleAgent:
    def __init__(self, config):
        self.config = config
        self.llm = ChatOpenAI(
            **config['llm'],
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        self.prompt = ChatPromptTemplate.from_template(config['prompt_template'])
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    @staticmethod
    def format_group(group_dict):
        parts = [f"{count} {character}" 
                 for character, count in group_dict.items() 
                 if count > 0]
        total = sum(group_dict.values())
        return " + ".join(parts) + f" = {total} life in total"
    
    @staticmethod
    def format_group_csv(group_dict):
        parts = [f"{count} {character}" 
                 for character, count in group_dict.items() 
                 if count > 0]
        total = sum(group_dict.values())
        return " ".join(parts) + f" ({total} total)"
    
    async def analyze(self, scenario, start_time):
        response = await self.chain.ainvoke({
            "scenario_type": scenario.type,
            "legal_status": scenario.legalStatus,
            "left_desc": self.format_group(scenario.left),
            "right_desc": self.format_group(scenario.right)
        })
        
        try:
            response = response.strip()
            response = response.replace('```json', '').replace('```', '')
            parsed_response = json.loads(response)
            
            runtime = round(time.time() - start_time, 4)
            parsed_response['runtime'] = f"{runtime}s"
            
            # Save to CSV
            self._save_to_csv(scenario, parsed_response, runtime)
            
            return parsed_response
            
        except json.JSONDecodeError as e:
            runtime = round(time.time() - start_time, 4)
            error_response = {
                "decision": "ERROR",
                "reason": f"Failed to parse AI response: {str(e)}",
                "runtime": f"{runtime}s"
            }
            
            # Save error to CSV
            self._save_to_csv(scenario, error_response, runtime)
            
            return error_response
    
    def _save_to_csv(self, scenario, response, runtime):
        csv_file = 'scenario_responses.csv'
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'model', 'temperature', 'scenario_type', 'legal_status', 
                               'left_group', 'right_group', 'decision', 'reason', 'runtime'])
            
            writer.writerow([
                datetime.now().isoformat(),
                self.config['llm']['model'],
                self.config['llm']['temperature'],
                scenario.type,
                scenario.legalStatus,
                self.format_group_csv(scenario.left),
                self.format_group_csv(scenario.right),
                response['decision'],
                response['reason'],
                runtime
            ])
