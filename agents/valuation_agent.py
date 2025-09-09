from crewai import Agent
from crewai.tools import tool
import pandas as pd
import numpy as np
from prompts import AgentPrompts
from .base_agent import BaseAgent


class ValuationAgent(BaseAgent):
    def __init__(self, llm, **kwargs):
        super().__init__(llm, **kwargs)

    def _get_tools(self) -> list:
        @tool
        def getAnnualisedVolatilityTool(*args, **kwargs) -> str:
            """
            Get the annualised volatility for a company
            """
            df = pd.read_csv(
                "assets/Quote-Equity-RELIANCE-EQ-08-09-2024-to-08-09-2025.csv"
            )
            df["close "] = df["close "].str.replace(",", "").astype(float)
            log_returns = np.log(df["close "] / df["close "].shift(1))
            volatility = log_returns.std() * (252**0.5)
            return volatility

        @tool
        def getAnnualisedReturnTool(*args, **kwargs) -> float:
            """
            Get the annualised return for a company
            """
            df = pd.read_csv(
                "assets/Quote-Equity-RELIANCE-EQ-08-09-2024-to-08-09-2025.csv"
            )
            cummulative_return = (
                float(df["close "].iloc[-1].replace(",", ""))
                / float(df["close "].iloc[0].replace(",", ""))
            ) - 1
            annualised_return = (1 + cummulative_return) ** (252 / len(df)) - 1
            return annualised_return

        return [getAnnualisedVolatilityTool, getAnnualisedReturnTool]

    def get_agent_instance(self) -> Agent:
        return Agent(
            role="valuationAgent",
            goal=AgentPrompts.valuationAgentGoal.value,
            backstory=AgentPrompts.valuationAgentBackStory.value,
            tools=self._get_tools,
            verbose=True,
            memory=True,
            allow_delegation=False,
            llm=self.llm,
        ) 
