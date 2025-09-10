import sys
import warnings

from crew import InvestmentCrew

import streamlit as st
import os

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    # InvestmentCrew.stock = "RELIANCE.NS"
    inputs = {
        'topic': f'give me report for {InvestmentCrew.stock}',
    }
    return InvestmentCrew().crew().kickoff(inputs=inputs).raw


st.set_page_config(page_title="AlphaAgent", page_icon=":robot:")
st.title("AlphaAgents")

stock_input = st.text_input("Enter Stock Ticker")

financial_doc = st.file_uploader("Upload Your Financial Document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
if financial_doc is not None:
    filename = "uploadedfile_note.txt"
    filepath = os.path.join("assets/rag_assets", filename)
    with open(filepath, 'w') as f:
        f.write(financial_doc.getvalue().decode("utf-8"))

    st.success("Document uploaded successfully!")

button = st.button("Analyze")

if button:
    if stock_input:
        InvestmentCrew.stock = stock_input
        with st.spinner("Analyzing..."):
            st.markdown(f"""### Analyzing Stock Ticker: {InvestmentCrew.stock}
### Report: 
                        {run()}""")

    else:
        st.error("Please enter a stock ticker.")