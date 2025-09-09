from abc import ABC, abstractmethod
from crewai import Agent
from prompts import AgentPrompts


class BaseAgent(ABC):
    def __init__(self, llm, **kwargs):
        super().__init__()
        self.llm = llm
        self.kwargs = kwargs

    @abstractmethod
    def _get_tools(self) -> list:
        """Return a list of tools for the agent."""
        pass

    @abstractmethod
    def get_agent_instance(self) -> Agent:
        """Return an initialized crewAI Agent instance."""
        pass

