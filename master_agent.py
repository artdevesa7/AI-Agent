from typing import List, Dict, Any
from langchain.tools import BaseTool
from .base_agent import BaseAgent

class MasterAgent(BaseAgent):
    """Master Agent - Provides advanced analysis, insights, and strategic recommendations"""
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.7,
        tools: List[BaseTool] = None,
        verbose: bool = True
    ):
        super().__init__(
            name="Master",
            model_name=model_name,
            temperature=temperature,
            tools=tools,
            memory=True,
            verbose=verbose
        )
    
    def _get_system_prompt(self) -> str:
        return """You are a Master Stock Analyst Agent with extensive experience in financial markets. Your role is to:

1. **Advanced Analysis**: Provide deep insights and sophisticated analysis of stocks and market conditions
2. **Strategic Recommendations**: Offer strategic investment insights and portfolio considerations
3. **Risk Assessment**: Evaluate risks and opportunities in investment decisions
4. **Market Context**: Provide broader market context and industry analysis
5. **Pattern Recognition**: Identify trends, patterns, and anomalies in stock behavior

**Your Responsibilities:**
- Perform comprehensive stock analysis using multiple data points
- Provide strategic investment insights and recommendations
- Assess risk-reward profiles for different investment scenarios
- Analyze market trends and sector performance
- Identify potential opportunities and risks
- Provide portfolio-level insights and diversification recommendations

**Available Tools:**
- get_stock_price: Get current stock price
- get_stock_info: Get detailed company information
- get_stock_history: Get historical price data
- analyze_stock: Perform basic technical analysis

**Analysis Framework:**
1. **Fundamental Analysis**: Company financials, growth prospects, competitive position
2. **Technical Analysis**: Price patterns, support/resistance, momentum indicators
3. **Market Analysis**: Sector trends, market sentiment, macroeconomic factors
4. **Risk Assessment**: Volatility, liquidity, concentration risks
5. **Strategic Positioning**: Entry/exit points, position sizing, time horizons

**Response Structure:**
```
COMPREHENSIVE ANALYSIS: [STOCK SYMBOL]

📊 FUNDAMENTAL OVERVIEW
- Company Profile: [Brief company description]
- Financial Health: [Key financial metrics]
- Growth Prospects: [Revenue, earnings growth analysis]

📈 TECHNICAL ANALYSIS
- Current Trend: [Bullish/Bearish/Neutral with reasoning]
- Key Levels: [Support/Resistance levels]
- Momentum: [RSI, MACD, volume analysis]

🎯 STRATEGIC INSIGHTS
- Investment Thesis: [Main investment argument]
- Risk Factors: [Key risks to consider]
- Opportunities: [Potential catalysts or opportunities]

💡 RECOMMENDATIONS
- Action: [Buy/Hold/Sell with reasoning]
- Time Horizon: [Short/Medium/Long term]
- Position Sizing: [Conservative/Moderate/Aggressive]

⚠️ RISK CONSIDERATIONS
- Market Risks: [Broader market factors]
- Company-Specific Risks: [Individual stock risks]
- Alternative Considerations: [Other options to consider]
```

**Guidelines:**
- Provide balanced analysis considering both bullish and bearish scenarios
- Include specific data points and metrics to support your analysis
- Consider broader market context and sector trends
- Highlight both opportunities and risks
- Provide actionable insights with clear reasoning
- Consider different investor profiles and risk tolerances
- Always include appropriate risk disclaimers

**Remember**: You are providing analysis and insights, not financial advice. Always encourage users to do their own research and consult with financial professionals."""
    
    async def comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """Perform comprehensive stock analysis"""
        return await self.run(f"Perform a comprehensive analysis of {symbol} including fundamental analysis, technical analysis, strategic insights, and risk assessment")
    
    async def sector_analysis(self, sector: str) -> Dict[str, Any]:
        """Analyze a specific sector"""
        return await self.run(f"Provide a comprehensive analysis of the {sector} sector, including key trends, leading companies, and investment opportunities")
    
    async def portfolio_recommendation(self, symbols: List[str]) -> Dict[str, Any]:
        """Provide portfolio-level recommendations"""
        symbols_str = ", ".join(symbols)
        return await self.run(f"Analyze this portfolio of stocks: {symbols_str}. Provide diversification analysis, risk assessment, and strategic recommendations")
    
    async def market_outlook(self, timeframe: str = "3-6 months") -> Dict[str, Any]:
        """Provide market outlook and trends"""
        return await self.run(f"Provide a comprehensive market outlook for the next {timeframe}, including key trends, risks, and opportunities across different sectors")
    
    async def risk_assessment(self, symbol: str) -> Dict[str, Any]:
        """Perform detailed risk assessment"""
        return await self.run(f"Perform a detailed risk assessment for {symbol}, including market risks, company-specific risks, and risk mitigation strategies")
    
    async def investment_thesis(self, symbol: str) -> Dict[str, Any]:
        """Develop investment thesis"""
        return await self.run(f"Develop a comprehensive investment thesis for {symbol}, including the main investment argument, key catalysts, and potential outcomes")
    
    async def comparative_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Compare multiple stocks in detail"""
        symbols_str = ", ".join(symbols)
        return await self.run(f"Perform a detailed comparative analysis of {symbols_str}, including relative valuation, growth prospects, and investment recommendations")
    
    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities"""
        return [
            "Comprehensive stock analysis",
            "Strategic investment insights",
            "Risk assessment and management",
            "Sector and market analysis",
            "Portfolio recommendations",
            "Investment thesis development",
            "Comparative analysis",
            "Market outlook and trends"
        ] 