import sys
import warnings

from crew import InvestmentCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    InvestmentCrew.stock = "RELIANCE"
    inputs = {
        'topic': f'give me report for {InvestmentCrew.stock}',
    }
    InvestmentCrew().crew().kickoff(inputs=inputs)

run()