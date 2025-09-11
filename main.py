import sys
import warnings
import argparse
import os

from crew import InvestmentCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run(stock: str):
    """
    Run the crew for a given stock.
    """
    inputs = {
        'topic': f'give me report for {stock}',
    }
    return InvestmentCrew().crew().kickoff(inputs=inputs).raw


def main():
    parser = argparse.ArgumentParser(description="Run AlphaAgent for stock analysis.")
    parser.add_argument(
        "--stock",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. RELIANCE.NS)"
    )
    parser.add_argument(
        "--doc",
        type=str,
        help="Optional path to financial document (PDF, DOCX, TXT) for RAG"
    )

    args = parser.parse_args()

    # Set stock
    InvestmentCrew.stock = args.stock

    # Handle document if provided
    if args.doc:
        filename = "uploadedfile_note.txt"
        filepath = os.path.join("assets/rag_assets", filename)

        with open(args.doc, "rb") as f_in, open(filepath, "wb") as f_out:
            f_out.write(f_in.read())

        print(f"✅ Document {args.doc} saved to {filepath}")

    print(f"\n📊 Analyzing Stock Ticker: {args.stock}\n")
    print("### Report:\n")
    print(run(args.stock))


if __name__ == "__main__":
    main()
