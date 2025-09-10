from crewai import Agent, Crew, Process, LLM, Task
from crewai.project import CrewBase, agent, crew, task # type: ignore
from tavily import TavilyClient
import os
from crewai.tools import tool # type: ignore
from enum import Enum
import pandas as pd
import numpy as np
from openai import OpenAI

llm = LLM(
    model="openai/gpt-5-mini", 
    stop=["END"],
    seed=42
) 


class AgentPrompts(Enum):
    """
    This class contains backstories and prompts for all the agents defined in code
    """


    fundamentalAgentBackStory = """As a fundamental financial equity
analyst your primary responsibility is to analyze the most
recent 10K report provided for a company. You have access to a
powerful tool that can help you extract relevant information
from the 10K. Your analysis should be based solely on the
information that you retrieve using this tool. You can interact
with this tool using natural language queries. The tool will
understand your requests and return relevant text snippets
and data points from the 10K document. Keep checking if you
have answered the users’ question to avoid looping"""
 
    fundamentalAgentGoal = """To come up with position for a stock based on fundamental analysis. I will use the tools at your disposal to help formulate a final position"""

    valuationAgentBackStory = """As a valuation equity analyst, your primary responsibility is to analyze the valuation trends of a given asset or portfolio over an extended time horizon. To complete the task, you must analyze the historical valuation data of the asset or portfolio provided, 
        identify trends and patterns in valuation metrics over time, and interpret the implications of these trends for investors or stakeholders."""

    valuationAgentGoal = """To come up with position for a stock based on returns and volatility analysis. I will use the tools at your disposal to help formulate a final positiom"""


    sentimentAgentBackStory = """As a sentiment equity analyst your primary responsibility is to analyze the financial news, analyst
        ratings and disclosures related to the underlying security;
        and analyze its implication and sentiment for investors or
        stakeholders"""

    sentimentAgentGoal = """ To come up with position for a stock based on sentiment analysis. I will use the tools at your disposal to help formulate a final position"""





def get_tavily_search(stock = "reliance"):
    '''
        Get the tavily search results for a company
    '''
    tavily_client = TavilyClient(api_key="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")
    response = tavily_client.search("reliance news", max_results=3, topic="finance", search_depth="advanced", country="india", include_raw_content="markdown")
    return response['results']




@tool 
def getNewsBodyTool(*args, **kwargs) -> list:
    '''
        Get the news from tavily search and return only the revelant news with score > 0.6
    '''

    news_list = get_tavily_search("reliance")
    final_news_content = []

    for news in news_list:
        if news['score']>0.6:
            final_news_content.append(news['raw_content'])
            

    return final_news_content


@tool
def getAnnualisedVolatilityTool(*args, **kwargs) -> str:
    '''
        Get the annualised volatility for a company
    '''
    df = pd.read_csv('Quote-Equity-RELIANCE-EQ-08-09-2024-to-08-09-2025.csv')
    df['close '] = df['close '].str.replace(',', '').astype(float)
    log_returns = np.log(df['close '] / df['close '].shift(1))
    volatility = log_returns.std() * (252 ** 0.5)
    return volatility


@tool
def getAnnualisedReturnTool(*args, **kwargs) -> float:
    '''
        Get the annualised return for a company
    '''
    df = pd.read_csv('Quote-Equity-RELIANCE-EQ-08-09-2024-to-08-09-2025.csv')
    cummulative_return = (float(df['close '].iloc[-1].replace(',','')) / float(df['close '].iloc[0].replace(',',''))) - 1
    annualised_return = (1 + cummulative_return) ** (252 / len(df)) - 1
    return annualised_return

@tool
def fundamental_analysis_tool():
    '''
        Tool to analyze the financial report of a company and provide a summary
    '''

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    with open("INDAS_117298_1348254_16012025082021 (2).xml", "r", encoding="utf-8") as f:
        xml_content = f.read()


    response = client.chat.completions.create(
        model="gpt-5-nano",  
        messages=[
            {"role": "system", "content": "You are a financial analyst. You have to provide correct answers. IF you dont know the answer, you must say you dont know. "},
            {"role": "user", "content": f"Here is an XML file:\n{xml_content}\n\nSummarize the financial results. Dont ask for anything other prompt"}
        ],
    )
    return response.choices[0].message.content





valuationAgent = Agent(
    role="valuationAgent",
    goal=AgentPrompts.valuationAgentGoal.value,
    backstory=AgentPrompts.valuationAgentBackStory.value,
    tools=[getAnnualisedVolatilityTool, getAnnualisedReturnTool],
    verbose=True,
    memory=True,
    allow_delegation=False,
    llm=llm
) # type: ignore



sentimentAgent = Agent(
    role="sentimentAgent",
    goal=AgentPrompts.sentimentAgentGoal.value,
    backstory=AgentPrompts.sentimentAgentBackStory.value,
    tools=[getNewsBodyTool],
    verbose=True,
    memory=True,
    allow_delegation=False,
    llm=llm
) # type: ignore



fundamentalAgent = Agent(
    role="fundamentalAgent",
    goal=AgentPrompts.fundamentalAgentGoal.value,
    backstory=AgentPrompts.fundamentalAgentBackStory.value,
    tools=[fundamental_analysis_tool],
    verbose=True,
    memory=True,
    llm=llm
)


fundamental_task = Task(
   description="Analyze the company's 10K report using the fundamental analysis tool. Extract key financial information by querying the tool with specific questions about revenue trends, profitability metrics, debt levels, cash flow statements, segment performance, and risk factors. Use natural language queries to interact with the tool and gather comprehensive data from the XML document. Ensure you retrieve all necessary financial data points before formulating your investment recommendation.",
   expected_output="A comprehensive fundamental analysis report including: 1) Key financial metrics extracted from the 10K (revenue, net income, EPS, debt-to-equity ratio), 2) Year-over-year growth trends and financial performance analysis, 3) Segment-wise business performance breakdown, 4) Critical risk factors and their potential impact, 5) Cash flow analysis and liquidity position, 6) Final BUY/SELL/HOLD recommendation supported by specific data points from the 10K report.",
   agent=fundamentalAgent,
)


valuation_task = Task(
   description="Analyze Reliance stock's historical performance by calculating annualized returns and volatility. Use both metrics to assess risk-adjusted performance and provide a clear BUY, SELL, or HOLD recommendation with detailed reasoning.",
   expected_output="A comprehensive valuation report including: 1) Annualized return percentage, 2) Annualized volatility percentage, 3) Risk-return analysis, 4) Final BUY/SELL/HOLD recommendation with justification based on the metrics.",
   agent=valuationAgent,
)

sentiment_task = Task(
   description="Analyze recent news articles about Reliance to gauge market sentiment. Review the news content for positive or negative indicators about the company's performance, management decisions, and future prospects. Provide a sentiment-based investment recommendation.",
   expected_output="A sentiment analysis report including: 1) Summary of key news themes, 2) Overall sentiment assessment (positive/negative/neutral), 3) Key risks or opportunities identified, 4) Final BUY/SELL/HOLD recommendation based on news sentiment.",
   agent=sentimentAgent,
)


moderator = Agent(
    role="Investment Moderator",
    goal="Keep the investment debate structured, ask for responses, and finally give a concluding investment decision.",
    backstory="You are a neutral and fair investment debate moderator with deep market experience. "
              "You ensure that each Agent has the chance to present their case and that the debate remains productive. Each agent should provide with thier opinion on what the strategby should be and all of them must come toa final consensus that they agree on. ",
    verbose=True,
    allow_delegation=True,
    memory=True,  # type: ignore
    llm=llm
) # type: ignore



investment_debate_task = Task(
    description=(
        "use all the context available to you to moderate a debate between 2 investment agents given to you Sentiment Agent and Valuation Agent. "
        "Start a structured debate between the three investment analysts. "
        "Each agent should independently propose whether the stock should be a BUY, SELL, or HOLD. "
        "They must defend their recommendation with reasoning, challenge opposing recommendations, "
        "and respond to critiques. Ensure each analyst has at least 2 turns to speak. "
        "The goal is for the analysts to debate the merits of each position and work toward a consensus "
        "on the final investment recommendation."
    ),
    expected_output=(
        "A transcript of the investment debate with clear BUY/SELL/HOLD positions from each analyst, "
        "including reasoning, challenges, and responses."
    ),
    context=[valuation_task, sentiment_task],
    agent=moderator,  # Moderator orchestrates the discussion
) # type: ignore


conclusion_agent = Agent(
    role="Conclusion Agent",
    goal="You are to provide a final investment decision based on the debate moderated by the Investment Moderator.",
    backstory="You are a neutral and fair investment Analyst with deep market experience. "
              "After the debate, analyze the discussion carefully and provide a final investment decision. ",
    verbose=True,
    allow_delegation=False,
    memory=True,  # type: ignore
    llm=llm
) # type: ignore

investment_conclusion_task = Task(
    description=(
        "After the debate, analyze the discussion carefully and provide a final investment decision. "
        "The conclusion should summarize key arguments and declare the final recommendation: BUY, SELL, or HOLD "
        "with proper reasoning based on the strongest arguments presented."
    ),
    expected_output=(
        "A clear and reasoned final investment decision (BUY/SELL/HOLD) summarizing the debate "
        "and identifying which analytical approach provided the most compelling case."
    ),
    context=[investment_debate_task],
    agent=conclusion_agent,
) # type: ignore



crew = Crew(
    agents=[valuationAgent, sentimentAgent,fundamentalAgent, moderator, conclusion_agent],
    tasks=[ valuation_task,sentiment_task,fundamental_task,investment_debate_task, investment_conclusion_task],
    process=Process.sequential,
) 


result = crew.kickoff()
print("Final Investment Strategy Decision:\n", result)




