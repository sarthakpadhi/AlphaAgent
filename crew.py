from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool
from tavily import TavilyClient
from openai import OpenAI
import pandas as pd
import numpy as np
import os


# ------------------------
# LLMs
llm = LLM(model="openai/gpt-5-mini", stop=["END"], seed=42)


# ------------------------
# Tools
# ------------------------
def get_tavily_search(stock="reliance"):
    tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    response = tavily_client.search(
        f"{stock} news",
        max_results=3,
        topic="finance",
        search_depth="advanced",
        country="india",
        include_raw_content="markdown",
    )
    return response["results"]


@tool
def getNewsBodyTool(*args, **kwargs) -> list:
    """Get the news body for a company"""
    news_list = get_tavily_search("reliance")
    final_news_content = []
    for news in news_list:
        if news["score"] > 0.6:
            final_news_content.append(news["raw_content"])
    return final_news_content


@tool
def getAnnualisedVolatilityTool(*args, **kwargs) -> str:
    """Get the annualised volatility for a company"""
    df = pd.read_csv("assets/Quote-Equity-RELIANCE-EQ-08-09-2024-to-08-09-2025.csv")
    df["close "] = df["close "].str.replace(",", "").astype(float)
    log_returns = np.log(df["close "] / df["close "].shift(1))
    volatility = log_returns.std() * (252**0.5)
    return volatility


@tool
def getAnnualisedReturnTool(*args, **kwargs) -> float:
    """Get the annualised return for a company"""
    df = pd.read_csv("assets/Quote-Equity-RELIANCE-EQ-08-09-2024-to-08-09-2025.csv")
    cummulative_return = (
        float(df["close "].iloc[-1].replace(",", ""))
        / float(df["close "].iloc[0].replace(",", ""))
    ) - 1
    annualised_return = (1 + cummulative_return) ** (252 / len(df)) - 1
    return annualised_return


@tool
def fundamental_analysis_tool():
    """Tool to analyze the financial report of a company and provide a summary"""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    with open(
        "assets/INDAS_117298_1348254_16012025082021 (2).xml", "r", encoding="utf-8"
    ) as f:
        xml_content = f.read()

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {
                "role": "system",
                "content": "You are a financial analyst. Provide correct answers only. If you don't know, say so.",
            },
            {
                "role": "user",
                "content": f"Here is an XML file:\n{xml_content}\n\nSummarize the financial results. Don't ask for anything else.",
            },
        ],
    )
    return response.choices[0].message.content


# ------------------------
# CrewBase Class
# ------------------------
@CrewBase
class InvestmentCrew:
    """Investment Crew for Stock Analysis & Debate"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    llm = llm

    # -------- Agents --------
    @agent
    def fundamental_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["fundamental_analyst"],
            tools=[fundamental_analysis_tool],
            llm=self.llm,
        )

    @agent
    def valuation_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["valuation_analyst"],
            tools=[getAnnualisedVolatilityTool, getAnnualisedReturnTool],
            llm=self.llm,
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
