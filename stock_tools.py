import yfinance as yf
import requests
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import json

class StockPriceInput(BaseModel):
    symbol: str = Field(description="Stock symbol (e.g., AAPL, MSFT, GOOGL)")

class StockInfoInput(BaseModel):
    symbol: str = Field(description="Stock symbol to get information about")

class StockHistoryInput(BaseModel):
    symbol: str = Field(description="Stock symbol")
    period: str = Field(default="1mo", description="Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)")

class StockAnalysisInput(BaseModel):
    symbol: str = Field(description="Stock symbol to analyze")

class StockPriceTool(BaseTool):
    name = "get_stock_price"
    description = "Get the current stock price for a given symbol"
    args_schema = StockPriceInput

    def _run(self, symbol: str) -> str:
        """Get current stock price"""
        try:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            current_price = info.get('currentPrice', info.get('regularMarketPrice'))
            
            if current_price:
                return f"Current price of {symbol.upper()}: ${current_price:.2f}"
            else:
                return f"Could not retrieve price for {symbol.upper()}"
        except Exception as e:
            return f"Error getting stock price for {symbol}: {str(e)}"

class StockInfoTool(BaseTool):
    name = "get_stock_info"
    description = "Get detailed information about a stock including company name, sector, market cap, etc."
    args_schema = StockInfoInput

    def _run(self, symbol: str) -> str:
        """Get stock information"""
        try:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            
            relevant_info = {
                'Company Name': info.get('longName', 'N/A'),
                'Sector': info.get('sector', 'N/A'),
                'Industry': info.get('industry', 'N/A'),
                'Market Cap': f"${info.get('marketCap', 0):,}" if info.get('marketCap') else 'N/A',
                'Current Price': f"${info.get('currentPrice', 0):.2f}" if info.get('currentPrice') else 'N/A',
                '52 Week High': f"${info.get('fiftyTwoWeekHigh', 0):.2f}" if info.get('fiftyTwoWeekHigh') else 'N/A',
                '52 Week Low': f"${info.get('fiftyTwoWeekLow', 0):.2f}" if info.get('fiftyTwoWeekLow') else 'N/A',
                'P/E Ratio': info.get('trailingPE', 'N/A'),
                'Dividend Yield': f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else 'N/A'
            }
            
            result = f"Information for {symbol.upper()}:\n"
            for key, value in relevant_info.items():
                result += f"{key}: {value}\n"
            
            return result
        except Exception as e:
            return f"Error getting stock info for {symbol}: {str(e)}"

class StockHistoryTool(BaseTool):
    name = "get_stock_history"
    description = "Get historical stock data for analysis"
    args_schema = StockHistoryInput

    def _run(self, symbol: str, period: str = "1mo") -> str:
        """Get stock history"""
        try:
            ticker = yf.Ticker(symbol.upper())
            hist = ticker.history(period=period)
            
            if hist.empty:
                return f"No historical data found for {symbol.upper()}"
            
            latest = hist.iloc[-1]
            earliest = hist.iloc[0]
            
            result = f"Historical data for {symbol.upper()} ({period}):\n"
            result += f"Latest Close: ${latest['Close']:.2f}\n"
            result += f"Latest Volume: {latest['Volume']:,}\n"
            result += f"Period High: ${hist['High'].max():.2f}\n"
            result += f"Period Low: ${hist['Low'].min():.2f}\n"
            result += f"Price Change: ${latest['Close'] - earliest['Close']:.2f}\n"
            result += f"Percent Change: {((latest['Close'] - earliest['Close']) / earliest['Close'] * 100):.2f}%"
            
            return result
        except Exception as e:
            return f"Error getting stock history for {symbol}: {str(e)}"

class StockAnalysisTool(BaseTool):
    name = "analyze_stock"
    description = "Perform basic technical analysis on a stock"
    args_schema = StockAnalysisInput

    def _run(self, symbol: str) -> str:
        """Analyze stock"""
        try:
            ticker = yf.Ticker(symbol.upper())
            hist = ticker.history(period="3mo")
            
            if hist.empty:
                return f"No data available for analysis of {symbol.upper()}"
            
            # Calculate moving averages
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            hist['MA50'] = hist['Close'].rolling(window=50).mean()
            
            latest = hist.iloc[-1]
            current_price = latest['Close']
            ma20 = latest['MA20']
            ma50 = latest['MA50']
            
            # Basic analysis
            analysis = f"Technical Analysis for {symbol.upper()}:\n"
            analysis += f"Current Price: ${current_price:.2f}\n"
            analysis += f"20-day MA: ${ma20:.2f}\n"
            analysis += f"50-day MA: ${ma50:.2f}\n"
            
            # Trend analysis
            if current_price > ma20 and ma20 > ma50:
                analysis += "Trend: Bullish (Price above 20MA, 20MA above 50MA)\n"
            elif current_price < ma20 and ma20 < ma50:
                analysis += "Trend: Bearish (Price below 20MA, 20MA below 50MA)\n"
            else:
                analysis += "Trend: Mixed/Neutral\n"
            
            # Volatility
            volatility = hist['Close'].pct_change().std() * 100
            analysis += f"Volatility (3-month): {volatility:.2f}%\n"
            
            # Support and resistance levels
            support = hist['Low'].tail(20).min()
            resistance = hist['High'].tail(20).max()
            analysis += f"Recent Support: ${support:.2f}\n"
            analysis += f"Recent Resistance: ${resistance:.2f}"
            
            return analysis
        except Exception as e:
            return f"Error analyzing stock {symbol}: {str(e)}"

class AlphaVantageTool(BaseTool):
    name = "alpha_vantage_search"
    description = "Search for stocks using Alpha Vantage API"
    args_schema = StockPriceInput

    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def _run(self, symbol: str) -> str:
        """Search for stock using Alpha Vantage"""
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol.upper(),
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if 'Global Quote' in data:
                quote = data['Global Quote']
                return f"Alpha Vantage - {symbol.upper()}: ${float(quote.get('05. price', 0)):.2f}"
            else:
                return f"No data found for {symbol.upper()} via Alpha Vantage"
        except Exception as e:
            return f"Error with Alpha Vantage API: {str(e)}"

def get_stock_tools(api_key: Optional[str] = None) -> List[BaseTool]:
    """Get all stock-related tools"""
    tools = [
        StockPriceTool(),
        StockInfoTool(),
        StockHistoryTool(),
        StockAnalysisTool()
    ]
    
    if api_key:
        tools.append(AlphaVantageTool(api_key))
    
    return tools 