from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from .base_agent import BaseAgent
from .junior_agent import JuniorAgent
from .master_agent import MasterAgent
import asyncio
import json

class OrchestratorAgent(BaseAgent):
    """Orchestrator Agent - Coordinates between Junior and Master agents, manages workflow"""
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.3,
        tools: List[BaseTool] = None,
        verbose: bool = True
    ):
        super().__init__(
            name="Orchestrator",
            model_name=model_name,
            temperature=temperature,
            tools=tools,
            memory=True,
            verbose=verbose
        )
        
        # Initialize sub-agents
        self.junior_agent = None
        self.master_agent = None
        self.workflow_history = []
    
    def _get_system_prompt(self) -> str:
        return """You are an Orchestrator Agent responsible for coordinating a team of specialized stock analysis agents. Your role is to:

1. **Workflow Management**: Coordinate between Junior and Master agents based on task complexity
2. **Task Delegation**: Route requests to appropriate agents based on requirements
3. **Quality Control**: Ensure comprehensive and accurate responses
4. **User Interface**: Provide a unified interface for all stock analysis requests
5. **Process Optimization**: Streamline workflows and avoid redundant operations

**Agent Hierarchy:**
- **Junior Agent**: Basic data retrieval, simple analysis, factual information
- **Master Agent**: Advanced analysis, strategic insights, comprehensive recommendations
- **Orchestrator**: You coordinate and manage the workflow

**Delegation Strategy:**
- **Simple Queries** → Junior Agent (prices, basic info, simple comparisons)
- **Complex Analysis** → Master Agent (comprehensive analysis, strategic insights)
- **Multi-step Tasks** → Coordinate both agents in sequence
- **Quality Assurance** → Review and enhance responses as needed

**Workflow Patterns:**
1. **Data Gathering**: Junior Agent collects basic data
2. **Analysis**: Master Agent provides insights
3. **Synthesis**: You combine and present results
4. **Quality Check**: Ensure completeness and accuracy

**Response Guidelines:**
- Provide clear, structured responses
- Include agent attribution for different parts of analysis
- Ensure comprehensive coverage of user requests
- Maintain professional tone and formatting
- Include relevant disclaimers and context

**Example Response Structure:**
```
🤖 ORCHESTRATED ANALYSIS: [REQUEST SUMMARY]

📊 DATA GATHERING (Junior Agent)
[Basic data and facts]

🎯 ADVANCED ANALYSIS (Master Agent)
[Strategic insights and recommendations]

📋 SYNTHESIS & RECOMMENDATIONS
[Combined insights and final recommendations]

⚠️ IMPORTANT NOTES
[Disclaimers and additional context]
```

**Remember**: You are the conductor of the orchestra. Ensure smooth coordination, comprehensive coverage, and high-quality output."""
    
    def setup_agents(self, junior_tools: List[BaseTool], master_tools: List[BaseTool]):
        """Setup the junior and master agents"""
        self.junior_agent = JuniorAgent(tools=junior_tools, verbose=self.verbose)
        self.master_agent = MasterAgent(tools=master_tools, verbose=self.verbose)
    
    async def orchestrate_analysis(self, query: str) -> Dict[str, Any]:
        """Orchestrate analysis between agents based on query complexity"""
        try:
            # Determine query complexity and route accordingly
            complexity = self._assess_complexity(query)
            
            if complexity == "simple":
                return await self._handle_simple_query(query)
            elif complexity == "complex":
                return await self._handle_complex_query(query)
            else:
                return await self._handle_multi_step_query(query)
                
        except Exception as e:
            return {
                "agent": self.name,
                "input": query,
                "output": f"Error in orchestration: {str(e)}",
                "success": False
            }
    
    def _assess_complexity(self, query: str) -> str:
        """Assess the complexity of a query to determine routing"""
        query_lower = query.lower()
        
        # Simple queries
        simple_keywords = [
            "price", "current price", "stock price", "basic info", 
            "company info", "simple", "quick", "basic"
        ]
        
        # Complex queries
        complex_keywords = [
            "analysis", "comprehensive", "strategic", "recommendation",
            "investment thesis", "risk assessment", "portfolio", "sector",
            "market outlook", "comparative", "detailed"
        ]
        
        simple_count = sum(1 for keyword in simple_keywords if keyword in query_lower)
        complex_count = sum(1 for keyword in complex_keywords if keyword in query_lower)
        
        if complex_count > simple_count:
            return "complex"
        elif simple_count > 0 and complex_count == 0:
            return "simple"
        else:
            return "multi_step"
    
    async def _handle_simple_query(self, query: str) -> Dict[str, Any]:
        """Handle simple queries with Junior Agent"""
        if not self.junior_agent:
            return {"error": "Junior agent not initialized"}
        
        result = await self.junior_agent.run(query)
        
        return {
            "agent": self.name,
            "input": query,
            "output": f"🤖 JUNIOR AGENT ANALYSIS:\n\n{result['output']}",
            "sub_agent": "Junior",
            "success": result['success']
        }
    
    async def _handle_complex_query(self, query: str) -> Dict[str, Any]:
        """Handle complex queries with Master Agent"""
        if not self.master_agent:
            return {"error": "Master agent not initialized"}
        
        result = await self.master_agent.run(query)
        
        return {
            "agent": self.name,
            "input": query,
            "output": f"🎯 MASTER AGENT ANALYSIS:\n\n{result['output']}",
            "sub_agent": "Master",
            "success": result['success']
        }
    
    async def _handle_multi_step_query(self, query: str) -> Dict[str, Any]:
        """Handle multi-step queries using both agents"""
        if not self.junior_agent or not self.master_agent:
            return {"error": "Agents not initialized"}
        
        # Step 1: Get basic data with Junior Agent
        junior_result = await self.junior_agent.run(query)
        
        # Step 2: Get advanced analysis with Master Agent
        master_result = await self.master_agent.run(query)
        
        # Step 3: Synthesize results
        synthesis = await self._synthesize_results(junior_result, master_result, query)
        
        return {
            "agent": self.name,
            "input": query,
            "output": synthesis,
            "sub_agents": ["Junior", "Master"],
            "success": junior_result['success'] and master_result['success']
        }
    
    async def _synthesize_results(self, junior_result: Dict, master_result: Dict, original_query: str) -> str:
        """Synthesize results from multiple agents"""
        synthesis = f"🤖 ORCHESTRATED ANALYSIS: {original_query}\n\n"
        
        if junior_result['success']:
            synthesis += f"📊 DATA GATHERING (Junior Agent):\n{junior_result['output']}\n\n"
        
        if master_result['success']:
            synthesis += f"🎯 ADVANCED ANALYSIS (Master Agent):\n{master_result['output']}\n\n"
        
        synthesis += "📋 SYNTHESIS & RECOMMENDATIONS:\n"
        synthesis += "Based on the comprehensive analysis above, here are the key takeaways:\n"
        synthesis += "- Data accuracy verified by Junior Agent\n"
        synthesis += "- Strategic insights provided by Master Agent\n"
        synthesis += "- Combined analysis provides balanced perspective\n\n"
        
        synthesis += "⚠️ IMPORTANT NOTES:\n"
        synthesis += "- This analysis is for informational purposes only\n"
        synthesis += "- Always conduct your own research before making investment decisions\n"
        synthesis += "- Consider consulting with financial professionals\n"
        synthesis += "- Past performance does not guarantee future results"
        
        return synthesis
    
    async def get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """Get stock price using Junior Agent"""
        return await self.orchestrate_analysis(f"Get the current stock price for {symbol}")
    
    async def comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive analysis using Master Agent"""
        return await self.orchestrate_analysis(f"Perform comprehensive analysis of {symbol}")
    
    async def portfolio_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze portfolio using both agents"""
        symbols_str = ", ".join(symbols)
        return await self.orchestrate_analysis(f"Analyze this portfolio: {symbols_str}")
    
    async def market_research(self, query: str) -> Dict[str, Any]:
        """Conduct market research using appropriate agents"""
        return await self.orchestrate_analysis(query)
    
    def get_workflow_history(self) -> List[Dict[str, Any]]:
        """Get workflow execution history"""
        return self.workflow_history
    
    def clear_workflow_history(self):
        """Clear workflow history"""
        self.workflow_history = []
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            "orchestrator": {
                "name": self.name,
                "model": self.model_name,
                "tools": len(self.tools),
                "memory": bool(self.memory)
            },
            "junior": {
                "initialized": bool(self.junior_agent),
                "tools": len(self.junior_agent.tools) if self.junior_agent else 0
            },
            "master": {
                "initialized": bool(self.master_agent),
                "tools": len(self.master_agent.tools) if self.master_agent else 0
            }
        } 