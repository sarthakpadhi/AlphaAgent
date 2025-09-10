from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool
from crewai_tools import PDFSearchTool

from tavily import TavilyClient
from openai import OpenAI
import pandas as pd
import numpy as np
import os
import yfinance as yf

# ------------------------
# LLMs
llm = LLM(model="openai/gpt-5-mini", stop=["END"], seed=42)




# ------------------------
# Tools
# ------------------------
def get_tavily_search():
    tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    response = tavily_client.search(
        f"{InvestmentCrew.stock} news",
        max_results=3,
        topic="finance",
        search_depth="advanced",
        country="india",
        include_raw_content="markdown",
    )
    return response["results"]


@tool
def getNewsBodyTool() -> list:

    """
    Get the news body for a company
    args:
        stocks (str): Stock ticker
    """
    news_list = get_tavily_search(InvestmentCrew.stock)
    final_news_content = []
    for news in news_list:
        if news["score"] > 0.6:
            final_news_content.append(news["raw_content"])
    return final_news_content


@tool
def getAnnualisedVolatilityTool() -> str:

    """
    Get the annualised volatility for a company
    Args:
        stock (str): Stock ticker 
    """
    # df = pd.read_csv("assets/Quote-Equity-RELIANCE-EQ-08-09-2024-to-08-09-2025.csv")
    dat = yf.Ticker(f"{InvestmentCrew.stock}.NS")
    df = dat.history(period="3mo")
    log_returns = np.log(df["Close"] / df["Close "].shift(1))
    volatility = log_returns.std() * (252**0.5)
    return volatility


@tool
def getAnnualisedReturnTool() -> float:

    """
    Get the annualised return for a company
    Args:
        stock (str): Stock ticker
    """
    dat = yf.Ticker(f"{InvestmentCrew.stock}.NS")
    df = dat.history(period="3mo")
    cummulative_return = (
        float(df["close "].iloc[-1])
        / float(df["close "].iloc[0])
    ) - 1
    annualised_return = (1 + cummulative_return) ** (252 / len(df)) - 1
    return annualised_return


@tool
def fundamental_analysis_tool():
    """Tool to analyze the financial report of a company and provide a summary"""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    dat = yf.Ticker(f"{InvestmentCrew.stock}.NS")
    balance_sheet_data = dat.balance_sheet

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {
                "role": "system",
                "content": "You are a financial analyst. Provide correct answers only. If you don't know, say so.",
            },
            {
                "role": "user",
                "content": f"Here is an Pandas Dataframe:\n{balance_sheet_data}\n\nSummarize the financial results in INR. Don't ask for anything else.",
            },
        ],
    )
    return response.choices[0].message.content


RAGtool = PDFSearchTool(pdf='assets/25042025_Media_Release_RIL_Q4_FY2024_25_Financial_and_Operational_Performance.pdf')

# ------------------------
# CrewBase Class
# ------------------------
@CrewBase
class InvestmentCrew:

    """Investment Crew for Stock Analysis & Debate"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    stock ="RELIANCE"  # default stock

    llm = llm
    

    # -------- Agents --------
    @agent
    def fundamental_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["fundamental_analyst"],
            tools=[RAGtool],
            llm=self.llm,
        )

    @agent
    def valuation_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["valuation_analyst"],
            tools=[getAnnualisedVolatilityTool, getAnnualisedReturnTool],
            llm=self.llm,
        ) # type: ignore

    @agent
    def sentiment_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["sentiment_analyst"],
            tools=[getNewsBodyTool],
            llm=self.llm,
        ) # type: ignore

    @agent
    def moderator(self) -> Agent:
        return Agent(config=self.agents_config["moderator"], llm=self.llm) # type: ignore

    @agent
    def conclusion_agent(self) -> Agent:
        return Agent(config=self.agents_config["conclusion_agent"], llm=self.llm) # type: ignore

    # -------- Tasks --------
    @task
    def fundamental_task(self) -> Task:
        return Task(config=self.tasks_config["fundamental_task"]) # type: ignore

    @task
    def valuation_task(self) -> Task:
        return Task(config=self.tasks_config["valuation_task"]) # type: ignore

    @task
    def sentiment_task(self) -> Task:
        return Task(config=self.tasks_config["sentiment_task"]) # type: ignore

    @task
    def investment_debate_task(self) -> Task:
        return Task(config=self.tasks_config["investment_debate_task"]) # type: ignore

    @task
    def investment_conclusion_task(self) -> Task:
        return Task(config=self.tasks_config["investment_conclusion_task"]) # type: ignore

    @crew
    def crew(self) -> Crew:
        """Creates the full investment crew"""
        return Crew(
            agents=self.agents, # type: ignore
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
