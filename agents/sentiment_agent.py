from crewai import Agent
from crewai.tools import tool
import pandas as pd
import numpy as np
from prompts import AgentPrompts
from .base_agent import BaseAgent
from openai import OpenAI
from tavily import TavilyClient
import os


class SentimentAgent(BaseAgent):
    def __init__(self, llm, **kwargs):
        super().__init__(llm, **kwargs)

    def _get_tavily_search(stock="reliance"):
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = tavily_client.search(
            "reliance news",
            max_results=3,
            topic="finance",
            search_depth="advanced",
            country="india",
            include_raw_content="markdown",
        )
        return response["results"]

    def _get_tools(self) -> list:
        @tool
        def getNewsBodyTool(*args, **kwargs) -> list:
            """
            Get the news body for a company
            """
            news_list = self._get_tavily_search("reliance")
            final_news_content = []

            for news in news_list:
                if news["score"] > 0.6:
                    final_news_content.append(news["raw_content"])

            return final_news_content

        return [getNewsBodyTool]

    def get_agent_instance(self) -> Agent:
        return Agent(
            role="sentimentAgent",
            goal=AgentPrompts.sentimentAgentGoal.value,
            backstory=AgentPrompts.sentimentAgentBackStory.value,
            tools=self._get_tools,
            verbose=True,
            memory=True,
            allow_delegation=False,
            llm=self.llm,
        )
