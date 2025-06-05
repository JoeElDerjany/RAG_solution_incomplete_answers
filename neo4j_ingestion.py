import os
from dotenv import load_dotenv

load_dotenv()

from langchain_neo4j import Neo4jGraph
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer


AURA_INSTANCENAME = os.environ["AURA_INSTANCENAME"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")

llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o")
embeddings = OpenAIEmbeddings()
kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)

if __name__ == "__main__":
    print("Started")
    pdf_path = "Policies.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    print("Started splitting")
    docs = text_splitter.split_documents(documents=documents)
    print("Finished splitting")

    graph_transformer = LLMGraphTransformer(llm=llm)
    print("Started conversion")
    graph_docs = graph_transformer.convert_to_graph_documents(documents=docs)

    res = kg.add_graph_documents(graph_docs, include_source=True, baseEntityLabel=True)