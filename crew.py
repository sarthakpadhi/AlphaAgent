from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import yfinance as yf
import os

from tavily import TavilyClient
import pandas as pd
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

from typing import Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)  # root logger
logger = logging.getLogger(__name__)


# ------------------------
# LLMs
llm = LLM(model="openai/gpt-4o-mini", stop=["END"], seed=42)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
openai_llm = OpenAI(model="gpt-4o-mini")
client = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0,  # ensures deterministic outputs
)


# ------------------------
# Tools
# ------------------------


class SemanticChromaRAG:
    def __init__(self, docs_path: str, persist_directory: str = "./chroma_db"):
        loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents.")

        semantic_chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
        )
        semantic_chunks = semantic_chunker.create_documents(
            [d.page_content for d in documents]
        )
        print(f"Created {len(semantic_chunks)} semantic chunks.")

        self.vectordb = Chroma.from_documents(
            semantic_chunks, embedding=embeddings, persist_directory=persist_directory
        )
        self.vectordb.persist()

        self.retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "search_type": "mmr",
                "fetch_k": 10,
                "lambda_mult": 0.5,
            },
        )
        self.qa_chain = self.retriever | openai_llm

    def query(self, text: str):
        """Run a query over the semantic chunks using the QA chain."""
        return self.qa_chain.invoke(text)


# Lazy initialization - ChromaRAG will be initialized only when first needed
_chromarag: Optional[SemanticChromaRAG] = None

def get_chromarag(force_reinit: bool = False):
    """Get or create the ChromaRAG instance (lazy initialization)"""
    global _chromarag
    
    # Check if there are any documents in the directory
    docs_path = Path("assets/rag_assets/")
    if not docs_path.exists():
        print("No documents directory found")
        return None
    
    pdf_files = list(docs_path.glob("**/*.pdf"))
    if not pdf_files:
        print("No PDF documents found in assets/rag_assets/")
        return None
    
    if _chromarag is None or force_reinit:
        print("Initializing ChromaRAG...")
        _chromarag = SemanticChromaRAG(docs_path="assets/rag_assets/")
        print("ChromaRAG initialization complete")
    return _chromarag


@tool
def CustomRagTool(query: str) -> str:
    """
    Custom RAG tool to solve fundamental questions using uploaded documents.
    Use this for non balance sheet related questions.
    
    Args:
        query: The query to search in the vector store
    """
    # Get ChromaRAG instance only when the tool is actually used
    chromarag = get_chromarag()
    if chromarag is None:
        return "No documents available for analysis. Please upload documents first."
    
    try:
        results = chromarag.retriever.get_relevant_documents(query)
        ans = ""
        for result in results:
            ans += result.page_content
        return ans if ans else "No relevant information found in documents."
    except Exception as e:
        return f"Error searching documents: {str(e)}"


def get_tavily_search(stock_ticker, *args, **kwargs):
    tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    response = tavily_client.search(
        f"{stock_ticker} news",
        max_results=3,
        topic="finance",
        search_depth="advanced",
        country="india",
        include_raw_content="markdown",
    )
    return response["results"]


@tool
def getNewsBodyTool(*args, **kwargs) -> list:
    """
    Get the news body for a company
    args:
        stocks (str): Stock ticker
    """
    # Access the stock from the class variable
    stock = InvestmentCrew.stock
    print(f"[getNewsBodyTool] Using stock ticker: {stock}")
    news_list = get_tavily_search(stock)
    final_news_content = []
    for news in news_list:
        if news["score"] > 0.6:
            final_news_content.append(news["raw_content"])
    return final_news_content


@tool
def getAnnualisedVolatilityTool(*args, **kwargs) -> str:
    """
    Get the annualised volatility for a company
    Args:
        stock (str): Stock ticker
    """
    # Access the stock from the class variable
    stock = InvestmentCrew.stock
    print(f"[getAnnualisedVolatilityTool] Using stock ticker: {stock}")
    dat = yf.Ticker(f"{stock}")
    df = dat.history(period="3mo")
    log_returns = np.log(df["Close"] / df["Close"].shift(1))
    volatility = log_returns.std() * (252**0.5)
    return volatility


@tool
def getAnnualisedReturnTool(*args, **kwargs) -> float:
    """
    Get the annualised return for a company
    Args:
        stock (str): Stock ticker
    """
    # Access the stock from the class variable
    stock = InvestmentCrew.stock
    print(f"[getAnnualisedReturnTool] Using stock ticker: {stock}")
    dat = yf.Ticker(f"{stock}")
    df = dat.history(period="3mo")
    cummulative_return = (
        float(df["Close"].iloc[-1])
        / float(df["Close"].iloc[0])
    ) - 1
    annualised_return = (1 + cummulative_return) ** (252 / len(df)) - 1
    return annualised_return


@tool
def fundamental_analysis_tool(*args, **kwargs):
    """Tool to analyze the BalanceSheet of a company and provide a summary"""
    # Access the stock from the class variable
    stock = InvestmentCrew.stock
    print(f"[fundamental_analysis_tool] Using stock ticker: {stock}")
    # Get the stock balance sheet
    dat = yf.Ticker(f"{stock}")
    balance_sheet_data = dat.balance_sheet

    # Create messages
    messages = [
        SystemMessage(
            content="You are a financial analyst. Provide correct answers only. If you don't know, say so."
        ),
        HumanMessage(
            content=f"Here is a Pandas DataFrame:\n{balance_sheet_data}\n\nSummarize the financial results in INR. Don't ask for anything else."
        ),
    ]

    # Get response
    response = client(messages)

    # Access content
    summary = response.content
    return summary


# ------------------------
# CrewBase Class
# ------------------------
@CrewBase
class InvestmentCrew:
    """Investment Crew for Stock Analysis & Debate"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    llm = llm
    stock = None  # This will be set dynamically

    # -------- Agents --------
    @agent
    def fundamental_analyst(self) -> Agent:
        # Pass the tools as function references, not instances
        return Agent(
            config=self.agents_config["fundamental_analyst"],
            tools=[CustomRagTool, fundamental_analysis_tool],
            llm=self.llm,
        )

    @agent
    def valuation_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["valuation_analyst"],
            tools=[getAnnualisedVolatilityTool, getAnnualisedReturnTool],
            llm=self.llm,
            max_retry_limit= 3,
        )

    @agent
    def sentiment_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["sentiment_analyst"],
            tools=[getNewsBodyTool],
            llm=self.llm,
        )

    @agent
    def moderator(self) -> Agent:
        return Agent(config=self.agents_config["moderator"], llm=self.llm)

    @agent
    def conclusion_agent(self) -> Agent:
        return Agent(config=self.agents_config["conclusion_agent"], llm=self.llm)

    # -------- Tasks --------
    @task
    def fundamental_task(self) -> Task:
        return Task(config=self.tasks_config["fundamental_task"])

    @task
    def valuation_task(self) -> Task:
        return Task(config=self.tasks_config["valuation_task"])

    @task
    def sentiment_task(self) -> Task:
        return Task(config=self.tasks_config["sentiment_task"])

    @task
    def investment_debate_task(self) -> Task:
        return Task(config=self.tasks_config["investment_debate_task"])

    @task
    def investment_conclusion_task(self) -> Task:
        return Task(config=self.tasks_config["investment_conclusion_task"])

    @crew
    def crew(self) -> Crew:
        """Creates the full investment crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )