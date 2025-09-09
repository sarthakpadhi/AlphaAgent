from crewai import Agent, Crew, Process, LLM, Task
from crewai.project import CrewBase, agent, crew, task  # type: ignore
from tavily import TavilyClient
import os
from crewai.tools import tool  # type: ignore
from enum import Enum
import pandas as pd
import numpy as np
from openai import OpenAI
from prompts import AgentPrompts
from agents import ValuationAgent, SentimentAgent, FundamentalAgent


def analyseStock(llm: LLM) -> str:
    agentClasses = [ValuationAgent, SentimentAgent, FundamentalAgent]
    agentsList = [cls(llm).get_agent_instance() for cls in agentClasses]
    ##createCrew
    ##finalReturnResponse
    pass


def main():
    llm = LLM(model="openai/gpt-5-mini", stop=["END"], seed=42)
    analyseStock(llm)


if __name__ == "__main__":
    main()
