import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

load_dotenv()

template = """Context: {context}
Question: {question}

Generate an SQL query to answer the question based on the given context. If you need to make assumptions about table structures, please state them clearly.

SQL Query:"""

prompt = PromptTemplate.from_template(template)

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=256,
    temperature=0.2,
    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN'),
    timeout=120
)
llm_chain = prompt | llm

def generate_sql(context, question):
    return llm_chain.invoke({"context": context, "question": question})
