from crewai import Agent
from crewai.tools import tool
import pandas as pd
import numpy as np
from prompts import AgentPrompts
from .base_agent import BaseAgent
from openai import OpenAI


class FundamentalAgent(BaseAgent):
    def __init__(self, llm, **kwargs):
        super().__init__(llm, **kwargs)

    def _get_tools(self) -> list:
        @tool
        def fundamental_analysis_tool(*args, **kwargs):
            """
            Tool to analyze the financial report of a company and provide a summary
            """

            client = OpenAI()

            with open(
                "INDAS_117298_1348254_16012025082021 (2).xml", "r", encoding="utf-8"
            ) as f:
                xml_content = f.read()

            response = client.chat.completions.create(
                model="gpt-5-nano",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst. You have to provide correct answers. IF you dont know the answer, you must say you dont know. ",
                    },
                    {
                        "role": "user",
                        "content": f"Here is an XML file:\n{xml_content}\n\nSummarize the financial results. Dont ask for anything other prompt",
                    },
                ],
            )
            return response.choices[0].message.content

        return [fundamental_analysis_tool]

    def get_agent_instance(self) -> Agent:
        return Agent(
            role="fundamentalAgent",
            goal=AgentPrompts.fundamentalAgentGoal.value,
            backstory=AgentPrompts.fundamentalAgentBackStory.value,
            tools=self._get_tools(),
            verbose=True,
            memory=True,
            llm=self.llm,
        )
