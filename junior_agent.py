from typing import List, Dict, Any
from langchain.tools import BaseTool
from .base_agent import BaseAgent

class JuniorAgent(BaseAgent):
    """Junior Agent - Handles basic stock data retrieval and simple analysis"""
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.5,
        tools: List[BaseTool] = None,
        verbose: bool = True
    ):
        super().__init__(
            name="Junior",
            model_name=model_name,
            temperature=temperature,
            tools=tools,
            memory=True,
            verbose=verbose
        )
    
    def _get_system_prompt(self) -> str:
        return """You are a Junior Stock Analyst Agent. Your role is to:

1. **Data Retrieval**: Fetch current stock prices, basic company information, and historical data
2. **Simple Analysis**: Perform basic calculations and provide straightforward insights
3. **Data Presentation**: Present information in a clear, organized manner
4. **Error Handling**: Gracefully handle errors and provide helpful error messages

**Your Responsibilities:**
- Get current stock prices for requested symbols
- Retrieve basic company information (name, sector, market cap, etc.)
- Fetch historical data for specified time periods
- Perform basic technical analysis (moving averages, support/resistance)
- Present data in a user-friendly format

**Available Tools:**
- get_stock_price: Get current stock price
- get_stock_info: Get detailed company information
- get_stock_history: Get historical price data
- analyze_stock: Perform basic technical analysis

**Guidelines:**
- Always verify stock symbols before making requests
- Provide clear, concise responses
- Include relevant context with your data
- If you encounter errors, explain what went wrong and suggest alternatives
- Focus on factual data rather than investment advice
- Use appropriate formatting for numbers and percentages

**Example Response Format:**
```
Stock Analysis for [SYMBOL]:
- Current Price: $XXX.XX
- Company: [Company Name]
- Sector: [Sector]
- Key Metrics: [Relevant data]
- Technical Analysis: [Basic insights]
```

Remember: You are a data-focused agent. Provide accurate information and let users make their own investment decisions."""
    
    async def get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """Get current stock price"""
        return await self.run(f"Get the current stock price for {symbol}")
    
    async def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get detailed stock information"""
        return await self.run(f"Get detailed information about {symbol} including company name, sector, market cap, and key metrics")
    
    async def get_stock_history(self, symbol: str, period: str = "1mo") -> Dict[str, Any]:
        """Get historical stock data"""
        return await self.run(f"Get historical data for {symbol} for the past {period}")
    
    async def analyze_stock(self, symbol: str) -> Dict[str, Any]:
        """Perform basic stock analysis"""
        return await self.run(f"Perform a basic technical analysis for {symbol} including trend analysis, moving averages, and support/resistance levels")
    
    async def compare_stocks(self, symbols: List[str]) -> Dict[str, Any]:
        """Compare multiple stocks"""
        symbols_str = ", ".join(symbols)
        return await self.run(f"Compare the following stocks: {symbols_str}. Provide current prices, basic metrics, and a brief comparison")
    
    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities"""
        return [
            "Get current stock prices",
            "Retrieve company information",
            "Fetch historical data",
            "Perform basic technical analysis",
            "Compare multiple stocks",
            "Present data in organized format"
        ] 