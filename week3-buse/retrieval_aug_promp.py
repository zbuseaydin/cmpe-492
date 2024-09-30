from typing import List, Dict
from sentence_transformers import SentenceTransformer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_cohere import ChatCohere
from dotenv import load_dotenv

load_dotenv()

llm = ChatCohere(model="command-r-plus", temperature=0)
prompt_template = ChatPromptTemplate.from_messages(
    [("system", "{system_prompt}"), ("user", "{user_prompt}")]
)
chain = prompt_template | llm | StrOutputParser()
chat_history = []

class SentenceTransformersEmbeddings:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[Dict]):
        # Assuming documents are in dict format with a 'text' key for the content
        texts = [doc.get("text", "") for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def embed_query(self, query: str):
        embedding = self.model.encode([query])[0]
        return embedding.tolist()

chroma_client = Client(Settings())

# Define collection for roles
role_collection = chroma_client.create_collection(
    name="roles",
    embedding_function=DefaultEmbeddingFunction()
)

### Step 1: Store Role Descriptions in Chroma using Dict Format ###
# Define some example roles in dictionary format
roles = [
    {"name": "Technical Assistant", "description": "You are an expert in technical issues and help with coding or technical problem solving."},
    {"name": "Financial Advisor", "description": "You provide advice and insights on financial matters, including budgeting, investment, and expenses."},
    {"name": "Health Expert", "description": "You offer guidance on health, nutrition, and fitness-related topics."},
    {"name": "Mental Health Guide", "description": "You offer guidance on mental health, and psychology-related topics."},
    {"name": "Travel Guide", "description": "You provide travel tips, itinerary suggestions, and destination recommendations."},
]

# Embed the role descriptions and store them in Chroma
embedding_model = SentenceTransformersEmbeddings()
role_embeddings = embedding_model.embed_documents([{"text": role["description"]} for role in roles])

for role, embedding in zip(roles, role_embeddings):
    role_collection.add(
        documents=[role["description"]],
        metadatas=[{"name": role["name"]}],
        ids=[role["name"]],
        embeddings=[embedding]
    )

### Step 2: Function to Retrieve the Most Relevant Role from Chroma ###
def retrieve_relevant_role(user_input):
    user_embedding = embedding_model.embed_query(user_input)

    # Query Chroma for the most similar role
    results = role_collection.query(
        query_embeddings=[user_embedding],
        n_results=1  # Get the top 1 most relevant role
    )

    if results and results["metadatas"]:
        most_relevant_role = results["metadatas"][0][0]  # Get the metadata of the most relevant result
        return most_relevant_role["name"], results["documents"][0][0]
    return None

def generate_role_based_prompt(role_name, role_description, user_input):
    # Construct the prompt by combining role description and user input
    prompt = (
        f"Please respond to the following input considering your role.\n\n"
        f"User Input: {user_input}"
    )
    return prompt

### Function to Handle the Interaction ###
def handle_interaction(user_input):
    global chat_history
    # Retrieve the most relevant role from Chroma
    role_data = retrieve_relevant_role(user_input)
    if role_data:
        role_name, role_description = role_data
    else:
        role_name = "Default Assistant"
        role_description = "You are a helpful assistant capable of answering general queries."

    prompt = {"system_prompt": role_description, "user_prompt": user_input}
    response = chain.invoke(prompt)

    # Update chat history
    chat_history.append({"user": user_input, "assistant": response, "role": role_name})

    return {
        "chat_history": chat_history,
        "role": role_name,
        "role_description": role_description,
        "answer": response,
    }

if __name__ == "__main__":    
    user_input = input("User: ")

    while user_input.lower() != "stop":
        output = handle_interaction(user_input)
        print(f"{output['role']}: {output['answer']}")
        user_input = input("User: ")
