import asyncio
from typing import List, Dict, Any, Optional
from config import Config
from tools import get_stock_tools
from agents import OrchestratorAgent, JuniorAgent, MasterAgent

class StockAgentSystem:
    """Main system class for the AI Stock Analysis Agent System"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the stock agent system"""
        self.config = config or Config()
        self.config.validate_config()
        
        # Initialize tools
        self.stock_tools = get_stock_tools(self.config.get_stock_api_key())
        
        # Initialize agents
        self.orchestrator = None
        self.junior_agent = None
        self.master_agent = None
        
        # System state
        self.is_initialized = False
        self.session_history = []
        
        # Initialize the system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all agents and the system"""
        try:
            # Create orchestrator agent
            self.orchestrator = OrchestratorAgent(
                model_name=self.config.OPENAI_MODEL,
                temperature=self.config.ORCHESTRATOR_AGENT_TEMPERATURE,
                tools=self.stock_tools,
                verbose=self.config.VERBOSE
            )
            
            # Create junior and master agents
            self.junior_agent = JuniorAgent(
                model_name=self.config.OPENAI_MODEL,
                temperature=self.config.JUNIOR_AGENT_TEMPERATURE,
                tools=self.stock_tools,
                verbose=self.config.VERBOSE
            )
            
            self.master_agent = MasterAgent(
                model_name=self.config.OPENAI_MODEL,
                temperature=self.config.MASTER_AGENT_TEMPERATURE,
                tools=self.stock_tools,
                verbose=self.config.VERBOSE
            )
            
            # Setup orchestrator with sub-agents
            self.orchestrator.setup_agents(
                junior_tools=self.stock_tools,
                master_tools=self.stock_tools
            )
            
            self.is_initialized = True
            print("✅ Stock Agent System initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing system: {str(e)}")
            self.is_initialized = False
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Main method to analyze any stock-related query"""
        if not self.is_initialized:
            return {"error": "System not initialized"}
        
        try:
            # Use orchestrator to handle the query
            result = await self.orchestrator.orchestrate_analysis(query)
            
            # Add to session history
            self.session_history.append({
                "query": query,
                "result": result,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            return result
            
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "query": query
            }
    
    async def get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """Get current stock price"""
        return await self.analyze_query(f"Get the current stock price for {symbol}")
    
    async def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get detailed stock information"""
        return await self.analyze_query(f"Get detailed information about {symbol}")
    
    async def analyze_stock(self, symbol: str) -> Dict[str, Any]:
        """Perform comprehensive stock analysis"""
        return await self.analyze_query(f"Perform comprehensive analysis of {symbol}")
    
    async def compare_stocks(self, symbols: List[str]) -> Dict[str, Any]:
        """Compare multiple stocks"""
        symbols_str = ", ".join(symbols)
        return await self.analyze_query(f"Compare these stocks: {symbols_str}")
    
    async def portfolio_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze a portfolio of stocks"""
        symbols_str = ", ".join(symbols)
        return await self.analyze_query(f"Analyze this portfolio: {symbols_str}")
    
    async def market_research(self, topic: str) -> Dict[str, Any]:
        """Conduct market research"""
        return await self.analyze_query(f"Research the market for: {topic}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and information"""
        return {
            "initialized": self.is_initialized,
            "config": {
                "model": self.config.OPENAI_MODEL,
                "verbose": self.config.VERBOSE,
                "max_iterations": self.config.MAX_ITERATIONS
            },
            "agents": {
                "orchestrator": bool(self.orchestrator),
                "junior": bool(self.junior_agent),
                "master": bool(self.master_agent)
            },
            "tools": len(self.stock_tools),
            "session_history_length": len(self.session_history)
        }
    
    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get session history"""
        return self.session_history
    
    def clear_session_history(self):
        """Clear session history"""
        self.session_history = []
    
    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of each agent"""
        if not self.is_initialized:
            return {}
        
        return {
            "orchestrator": [
                "Coordinate between agents",
                "Route queries appropriately",
                "Synthesize results",
                "Quality control"
            ],
            "junior": self.junior_agent.get_capabilities() if self.junior_agent else [],
            "master": self.master_agent.get_capabilities() if self.master_agent else []
        }
    
    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get information about available tools"""
        return [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in self.stock_tools
        ]
    
    async def test_system(self) -> Dict[str, Any]:
        """Test the system with a simple query"""
        test_query = "Get the current stock price for AAPL"
        result = await self.analyze_query(test_query)
        
        return {
            "test_query": test_query,
            "result": result,
            "system_working": result.get("success", False)
        }
    
    def __str__(self) -> str:
        return f"StockAgentSystem(initialized={self.is_initialized}, agents=3, tools={len(self.stock_tools)})"
    
    def __repr__(self) -> str:
        return self.__str__() 