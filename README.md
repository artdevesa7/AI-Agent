# 🤖 AI Stock Analysis Agent System

A sophisticated multi-agent AI system built with LangChain for comprehensive stock market analysis. The system features three specialized agents working together to provide data-driven insights and strategic recommendations.

## 🏗️ Architecture

### Agent Hierarchy

1. **🎯 Orchestrator Agent** - The conductor that coordinates between agents
   - Routes queries based on complexity
   - Synthesizes results from multiple agents
   - Provides unified interface
   - Manages workflow and quality control

2. **📊 Junior Agent** - Data retrieval and basic analysis specialist
   - Gets current stock prices
   - Retrieves company information
   - Fetches historical data
   - Performs basic technical analysis
   - Presents data in organized format

3. **🎓 Master Agent** - Advanced analysis and strategic insights
   - Comprehensive stock analysis
   - Strategic investment insights
   - Risk assessment and management
   - Portfolio recommendations
   - Market outlook and trends

## 🛠️ Features

### Stock Data Tools
- **Real-time stock prices** using yfinance and Alpha Vantage
- **Company information** (sector, market cap, P/E ratio, etc.)
- **Historical data** with customizable time periods
- **Technical analysis** (moving averages, support/resistance, trends)
- **Multi-source data** with fallback mechanisms

### AI Agent Capabilities
- **Intelligent query routing** based on complexity
- **Multi-step analysis** combining data and insights
- **Memory and context** for conversation continuity
- **Error handling** and graceful degradation
- **Extensible architecture** for adding new tools

### User Interface
- **Interactive CLI** with command history
- **Batch processing** for multiple queries
- **Session management** with query history
- **System status** and health monitoring
- **Example demonstrations** and tutorials

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- (Optional) Alpha Vantage or Finnhub API key

### Installation

1. **Clone and navigate to the project:**
```bash
cd "AI Agent"
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
# Copy the example environment file
cp env_example.txt .env

# Edit .env with your API keys
# Required: OPENAI_API_KEY
# Optional: ALPHA_VANTAGE_API_KEY, FINNHUB_API_KEY
```

4. **Run the system:**
```bash
# Interactive mode
python main.py

# Run examples
python main.py examples

# Test the system
python main.py test
```

## 📖 Usage Examples

### Interactive Mode
```bash
python main.py
```

Then use commands like:
- `Get the current stock price for AAPL`
- `Analyze TSLA comprehensively`
- `Compare AAPL, MSFT, GOOGL`
- `Portfolio analysis: AAPL, MSFT, GOOGL, TSLA`

### Programmatic Usage
```python
import asyncio
from stock_agent_system import StockAgentSystem

async def main():
    # Initialize the system
    system = StockAgentSystem()
    
    # Get stock price
    result = await system.get_stock_price("AAPL")
    print(result["output"])
    
    # Comprehensive analysis
    result = await system.analyze_stock("TSLA")
    print(result["output"])
    
    # Compare stocks
    result = await system.compare_stocks(["AAPL", "MSFT", "GOOGL"])
    print(result["output"])

asyncio.run(main())
```

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Stock Agent System                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │  Orchestrator   │    │   User Query    │                │
│  │     Agent       │◄──►│   Interface     │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                                               │
│           ▼                                               │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   Junior Agent  │    │  Master Agent   │                │
│  │  (Data & Basic  │    │ (Advanced &     │                │
│  │   Analysis)     │    │  Strategic)     │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                       │                        │
│           ▼                       ▼                        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Stock Data Tools                           │ │
│  │  • yfinance (prices, info, history)                    │ │
│  │  • Alpha Vantage API                                   │ │
│  │  • Technical Analysis                                  │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Yes | - |
| `OPENAI_MODEL` | OpenAI model to use | No | `gpt-4` |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage API key | No | - |
| `FINNHUB_API_KEY` | Finnhub API key | No | - |
| `MASTER_AGENT_TEMPERATURE` | Master agent creativity | No | `0.7` |
| `JUNIOR_AGENT_TEMPERATURE` | Junior agent creativity | No | `0.5` |
| `ORCHESTRATOR_AGENT_TEMPERATURE` | Orchestrator creativity | No | `0.3` |
| `MAX_ITERATIONS` | Max agent iterations | No | `10` |
| `VERBOSE` | Enable verbose output | No | `True` |

### Agent Temperatures
- **Orchestrator (0.3)**: Conservative, focused on coordination
- **Junior (0.5)**: Balanced, good for data presentation
- **Master (0.7)**: Creative, good for strategic insights

## 🛠️ Available Tools

### Stock Data Tools
1. **get_stock_price** - Get current stock price
2. **get_stock_info** - Get detailed company information
3. **get_stock_history** - Get historical price data
4. **analyze_stock** - Perform basic technical analysis
5. **alpha_vantage_search** - Search using Alpha Vantage API

### Agent Methods
- `analyze_query()` - Main analysis method
- `get_stock_price()` - Get current price
- `get_stock_info()` - Get company info
- `analyze_stock()` - Comprehensive analysis
- `compare_stocks()` - Compare multiple stocks
- `portfolio_analysis()` - Portfolio analysis
- `market_research()` - Market research

## 📊 Example Outputs

### Simple Query (Junior Agent)
```
🤖 JUNIOR AGENT ANALYSIS:

Stock Analysis for AAPL:
- Current Price: $175.43
- Company: Apple Inc.
- Sector: Technology
- Key Metrics: Market Cap: $2.7T, P/E: 28.5
- Technical Analysis: Bullish trend, above 20MA
```

### Complex Query (Master Agent)
```
🎯 MASTER AGENT ANALYSIS:

COMPREHENSIVE ANALYSIS: TSLA

📊 FUNDAMENTAL OVERVIEW
- Company Profile: Tesla, Inc. - Electric vehicle and clean energy company
- Financial Health: Strong revenue growth, improving profitability
- Growth Prospects: Expanding global presence, new product lines

📈 TECHNICAL ANALYSIS
- Current Trend: Bullish with strong momentum
- Key Levels: Support at $200, Resistance at $250
- Momentum: RSI at 65, MACD positive

🎯 STRATEGIC INSIGHTS
- Investment Thesis: Long-term growth in EV market
- Risk Factors: Competition, regulatory changes
- Opportunities: New markets, technology advances

💡 RECOMMENDATIONS
- Action: Hold with accumulation on dips
- Time Horizon: Long-term (2-5 years)
- Position Sizing: Moderate allocation

⚠️ RISK CONSIDERATIONS
- Market Risks: Economic cycles, interest rates
- Company-Specific Risks: Execution, competition
- Alternative Considerations: Diversify across EV sector
```

## 🔍 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Ensure `OPENAI_API_KEY` is set in `.env`
   - Verify the key is valid and has sufficient credits

2. **Stock Data Not Available**
   - System uses yfinance as fallback if no API keys
   - Some stocks may have limited data availability

3. **Agent Initialization Failed**
   - Check all required dependencies are installed
   - Verify environment variables are correctly set

### Debug Mode
Set `VERBOSE=True` in your `.env` file to see detailed agent interactions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with API terms of service and applicable regulations.

## ⚠️ Disclaimer

This system provides analysis and insights for informational purposes only. It does not constitute financial advice. Always conduct your own research and consult with financial professionals before making investment decisions. Past performance does not guarantee future results.

## 🎯 Roadmap

- [ ] Web interface with Streamlit/FastAPI
- [ ] Real-time market alerts
- [ ] Portfolio tracking and rebalancing
- [ ] Advanced technical indicators
- [ ] Sentiment analysis integration
- [ ] Backtesting capabilities
- [ ] Multi-language support
- [ ] Mobile app integration

---

**Built with ❤️ using LangChain and OpenAI** 