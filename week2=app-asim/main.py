from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os
from datetime import datetime
import logging

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


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    llm = ChatCohere(model="command-r-plus")
    prompt_template = ChatPromptTemplate.from_messages(
        [("user", "Give the answer of the question: {question}.\nContext: {context}")]
    )

    directory_loader = DirectoryLoader(
        "./sliced",  
        glob="**/*.txt",    # Only load text files
        loader_cls=TextLoader
    )
    docs = directory_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embedding_model = SentenceTransformersEmbeddings()

    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model, persist_directory=None)
    retriever = vectorstore.as_retriever()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # Set up logging
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'rag_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

    question = input("Enter your question: ")

    while question != "-1":
        result = rag_chain.invoke({"question": question})
        logging.info(f"Question: {question}\nAnswer: {result}\n")
        question = input("Enter your question: ")

    print(f"Log file created: {log_file}")
