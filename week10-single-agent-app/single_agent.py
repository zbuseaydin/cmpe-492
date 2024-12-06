import json
import time
import asyncio
import pdfplumber
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
import formatter_methods as formatter

load_dotenv()


class SentenceTransformersEmbeddings:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents):
        texts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        if embeddings is None or len(embeddings) == 0:
            return []
        return embeddings.tolist()

    def embed_query(self, query):
        embedding = self.model.encode([query])[0]
        return embedding.tolist()
    

class PDFTextLoader:
    def __init__(self, filepath=None):
        # You can store the loader_config or pass it if needed
        self.filepath = filepath

    def load(self):
        """Extract text from a PDF and return it as a Document object."""
        with pdfplumber.open(self.filepath) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        
        # Return the text as a Document object
        return [Document(page_content=text)]

    def lazy_load(self):
        """Lazy load method for PDF files."""
        return self.load()  # Simply call load with the filepath


class SingleAgent:
    def __init__(self, config):
        self.config = config
        self.use_rag = config.get('use_rag', False)  # Get the flag from config, default to True
        
        # Set up the LLM model with callbacks
        self.llm = ChatOpenAI(
            **config['llm'],
            callbacks=[StreamingStdOutCallbackHandler()]
        )

        # Initialize the prompt template
        self.agent_attributes = config['agent_attributes']
        
        # Setup vector store and retriever if RAG is enabled
        if self.use_rag:
            self._setup_vector_store()


    def _setup_vector_store(self):
        """Set up the vector store and retriever."""
        directory_loader = DirectoryLoader(
            "./ethics-documents",  # Directory with your ethics-related documents
            glob="**/*.pdf",       # Load PDF files only
            loader_cls=PDFTextLoader  # Use the custom PDF loader
        )
        docs = directory_loader.load()

        # Split documents into chunks for processing
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Initialize the embedding model and vector store
        embedding_model = SentenceTransformersEmbeddings()
        self.vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model, persist_directory=None)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 docs

    
    async def analyze(self, scenario, start_time):
        """Analyze the scenario using the RAG chain or non-RAG model."""

        input_dict = {
            "left_desc": formatter.format_group(scenario.left),
            "right_desc": formatter.format_group(scenario.right),
            "agent_role": self.agent_attributes["role"],
            "agent_gender": self.agent_attributes["gender"],
            "agent_age": self.agent_attributes["age"],
            "agent_education_level": self.agent_attributes["education_level"],
            "agent_calmness": self.agent_attributes["calmness"],
            "agent_empathy": self.agent_attributes["empathy"],
            "agent_analytical_thinking": self.agent_attributes["analytical_thinking"],
            "agent_risk_tolerance": self.agent_attributes["risk_tolerance"],
            "agent_decisiveness": self.agent_attributes["decisiveness"]
        }

        if self.use_rag:
            prompt = PromptTemplate(
                template=config['prompt_template_with_role'],
                input_variables=["context", "question"],
                partial_variables=input_dict,
                output_parser=StrOutputParser()
            )
            rag_chain = {"context": self.retriever | formatter.format_docs, "question": RunnablePassthrough()} | prompt | self.llm | StrOutputParser()
            response = await rag_chain.ainvoke({"question": "What is relevant to the ethical concerns while making a decision for moral machine experiment?"})
        else:
            prompt = ChatPromptTemplate.from_template(config['prompt_template_with_role'])
            chain = (prompt | self.llm | StrOutputParser())
            response = await chain.ainvoke(input_dict)
        
        try:
            response = response.strip()
            response = response.replace('```json', '').replace('```', '')
            parsed_response = json.loads(response)
            
            runtime = round(time.time() - start_time, 4)
            parsed_response['runtime'] = f"{runtime}s"
            
            # Save to CSV
            formatter._save_to_csv(scenario, parsed_response, runtime, self.agent_attributes, self.config)
            
            return parsed_response
            
        except json.JSONDecodeError as e:
            runtime = round(time.time() - start_time, 4)
            error_response = {
                "decision": "ERROR",
                "reason": f"Failed to parse AI response: {str(e)}",
                "runtime": f"{runtime}s"
            }
            
            # Save error to CSV
            formatter._save_to_csv(scenario, error_response, runtime)
            
            return error_response
    


if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Create an instance of the SingleAgent class
    agent = SingleAgent(config)

    # Example scenario (replace this with an actual scenario class or data structure)
    class Scenario:
        def __init__(self, scenario_type, legal_status, left, right):
            self.type = scenario_type
            self.legalStatus = legal_status
            self.left = left
            self.right = right

    # Example scenario data (modify according to your needs)
    scenario = Scenario(
        scenario_type="-",
        legal_status=None,
        left={"Pregnant Woman": 2},
        right={"Baby": 2}
    )

    # Start time to measure runtime
    start_time = time.time()

    # Analyze the scenario
    response = asyncio.run(agent.analyze(scenario, start_time))
    # Output the result
    print(response)
