# AlphaAgent
An implementation of the AlphaAgent paper published by BlackRock. 
Instead of AutoGen for orchestrating debates which is used in the original paper, we used crewAI as it is easier to onboard new Agents.
https://arxiv.org/abs/2508.11152



## 🎯 Features

- **Multi-Agent Analysis**: Three specialized agents analyze different aspects of investment
- **RAG Integration**: Semantic search through financial documents using ChromaDB
- **Real-time Data**: Yahoo Finance integration for live market data
- **News Sentiment**: Advanced news analysis using Tavily API
- **Structured Debate**: Moderated discussion between agents for consensus building
- **Comprehensive Reports**: Detailed analysis with clear BUY/SELL/HOLD recommendations

## 🏗️ Architecture

### Agents
1. **Fundamental Analyst** - Analyzes financial reports and balance sheets
2. **Valuation Analyst** - Calculates returns, volatility, and risk metrics
3. **Sentiment Analyst** - Processes news and market sentiment
4. **Moderator** - Facilitates structured debate between analysts
5. **Conclusion Agent** - Provides final investment decision

### Tools
- **CustomRagTool** - RAG-based financial document analysis that uses SemanticChunking
- **fundamentalAnalysisTool** - Balance sheet analysis
- **getAnnualisedVolatilityTool** - Volatility calculation
- **getAnnualisedReturnTool** - Return analysis
- **getNewsBodyTool** - News content extraction

## 📋 Prerequisites

- Python 3.8+
- UV package manager
- OpenAI API key
- Tavily API key

## 🚀 Installation

1. **Install dependencies using UV**
   ```bash
   uv sync
   ```

2. **Set up environment variables**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
   echo "TAVILY_API_KEY=your_tavily_api_key_here" >> .env
   ```


## ⚙️ Configuration

### Agent Configuration (config/agents.yaml)
The agents are configured with specific roles, goals, and backstories as defined in the provided YAML structure.

### Task Configuration (config/tasks.yaml)
Tasks define the workflow and expected outputs for each analysis phase.

## 🎮 Usage

### Running with UV
```bash
# Activate the virtual environment
uv run python main.py --stock <ticker> --doc </path/to/your/pdfdocs>
```
Alternatively, you can directly put multiple docs in the "assets/rag_assets" folder

## 📊 Analysis Workflow

1. **Fundamental Analysis**
   - Extracts financial data from uploaded documents
   - Analyzes balance sheet using Yahoo Finance
   - Provides comprehensive financial metrics

2. **Valuation Analysis**
   - Calculates 3-month annualized returns
   - Computes volatility metrics
   - Performs risk-return analysis

3. **Sentiment Analysis**
   - Fetches recent news articles
   - Analyzes market sentiment
   - Identifies key opportunities and risks

4. **Moderated Debate**
   - Structured discussion between analysts
   - Challenges and defenses of positions
   - Consensus building process

5. **Final Conclusion**
   - Synthesizes all analyses
   - Provides clear investment recommendation
   - Justifies decision with evidence

## 📝 Output Format

Each analysis phase produces structured reports including:
- Key metrics and findings
- Risk assessments
- Clear BUY/SELL/HOLD recommendations
- Supporting evidence and reasoning


## 🛠️ Adding New Agents and Tools

### Adding a New Agent

1. **Define the Agent in YAML Configuration**
   
   Add to `config/agents.yaml`:
   ```yaml
   technical_analyst:
     role: >
       Technical Analysis Specialist
     goal: >
       Analyze price patterns, technical indicators, and chart formations
       to provide trading insights and price predictions.
     backstory: >
       You are an expert technical analyst with deep knowledge of chart
       patterns, technical indicators, and market psychology. You use
       historical price data to identify trends and potential entry/exit points.
   ```

2. **Create the Agent Method**
   
   Add to your `InvestmentCrew` class:
   ```python
   @agent
   def technical_analyst(self) -> Agent:
       return Agent(
           config=self.agents_config["technical_analyst"],
           tools=[your_technical_tools],  # Add relevant tools
           llm=self.llm,
       )
   ```

3. **Update the Crew Configuration**
   
   Add the new agent to your crew:
   ```python
   @crew
   def crew(self) -> Crew:
       return Crew(
           agents=[
               self.fundamental_analyst(),
               self.valuation_analyst(),
               self.sentiment_analyst(),
               self.technical_analyst(),  # New agent
               self.moderator(),
               self.conclusion_agent()
           ],
           tasks=self.tasks,
           process=Process.sequential,
           verbose=True,
       )
   ```

### Adding a New Tool

#### Method 1: Function-based Tool (Recommended for simple tools)

```python
@tool
def get_technical_indicators(*args, **kwargs) -> dict:
    """
    Calculate technical indicators for a stock
    Args:
        period (str): Time period for analysis (e.g., '1mo', '3mo', '1y')
    Returns:
        dict: Technical indicators including RSI, MACD, Bollinger Bands
    """
    import talib
    
    # Get stock data
    ticker = yf.Ticker(f"{InvestmentCrew.stock}.NS")
    df = ticker.history(period=kwargs.get('period', '3mo'))
    
    # Calculate indicators
    indicators = {
        'rsi': talib.RSI(df['Close'].values),
        'macd': talib.MACD(df['Close'].values),
        'bb_upper': talib.BBANDS(df['Close'].values)[0],
        'bb_lower': talib.BBANDS(df['Close'].values)[2],
    }
    
    return indicators
```

#### Method 2: Class-based Tool (For complex tools with state)

```python
class TechnicalAnalysisInput(BaseModel):
    """Input schema for Technical Analysis Tool."""
    indicator: str = Field(..., description="Technical indicator to calculate (rsi, macd, bollinger)")
    period: str = Field(default="3mo", description="Time period for analysis")

class TechnicalAnalysisTool(BaseTool):
    name: str = "TechnicalAnalysisTool"
    description: str = "Calculate and analyze technical indicators for stock analysis"
    args_schema: Type[BaseModel] = TechnicalAnalysisInput
    
    def _run(self, indicator: str, period: str = "3mo") -> str:
        """Execute technical analysis"""
        try:
            ticker = yf.Ticker(f"{InvestmentCrew.stock}.NS")
            df = ticker.history(period=period)
            
            if indicator.lower() == "rsi":
                import talib
                rsi = talib.RSI(df['Close'].values)
                current_rsi = rsi[-1]
                return f"Current RSI: {current_rsi:.2f} - {'Overbought' if current_rsi > 70 else 'Oversold' if current_rsi < 30 else 'Neutral'}"
            
            # Add more indicators as needed
            return f"Indicator {indicator} calculated successfully"
            
        except Exception as e:
            return f"Error calculating {indicator}: {str(e)}"
```

### Adding a New Task

1. **Define Task in YAML Configuration**
   
   Add to `config/tasks.yaml`:
   ```yaml
   technical_analysis_task:
     description: >
       Perform comprehensive technical analysis on the stock using various
       indicators including RSI, MACD, Bollinger Bands, and moving averages.
       Identify key support and resistance levels, trend direction, and
       potential entry/exit points.
     expected_output: >
       A technical analysis report including:
         1) Current trend direction and strength,
         2) Key technical indicators (RSI, MACD, Bollinger Bands),
         3) Support and resistance levels,
         4) Technical BUY/SELL/HOLD recommendation with price targets.
     agent: technical_analyst
   ```

2. **Create Task Method**
   
   Add to your `InvestmentCrew` class:
   ```python
   @task
   def technical_analysis_task(self) -> Task:
       return Task(config=self.tasks_config["technical_analysis_task"])
   ```

3. **Update Dependent Tasks**
   
   Add context to tasks that should use this analysis:
   ```yaml
   investment_debate_task:
     # ... existing configuration ...
     context: [valuation_task, sentiment_task, fundamental_task, technical_analysis_task]
   ```

### Tool Integration Examples

#### External API Tool
```python
@tool
def get_insider_trading_data(*args, **kwargs) -> str:
    """Fetch insider trading information for the stock"""
    import requests
    
    # Example API call (replace with actual API)
    response = requests.get(f"https://api.example.com/insider/{InvestmentCrew.stock}")
    return response.json()
```

#### Database Tool
```python
class DatabaseQueryTool(BaseTool):
    name: str = "DatabaseQueryTool"
    description: str = "Query internal database for historical analysis data"
    
    def _run(self, query: str) -> str:
        """Execute database query"""
        # Connect to your database
        # Execute query
        # Return results
        pass
```

#### File Processing Tool
```python
@tool
def process_earnings_transcript(*args, **kwargs) -> str:
    """Process and analyze earnings call transcripts"""
    # Read transcript files
    # Perform NLP analysis
    # Extract key insights
    pass
```


## 🔧 Dependencies

Key packages managed by UV:
- `crewai` - Multi-agent framework
- `langchain` - LLM orchestration
- `yfinance` - Yahoo Finance data
- `tavily-python` - News API
- `chromadb` - Vector database
- `pandas` - Data manipulation
- `numpy` - Numerical computations
