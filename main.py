#!/usr/bin/env python3
"""
AI Stock Analysis Agent System
Main application file with command-line interface and examples
"""

import asyncio
import json
import sys
from typing import List, Dict, Any
from stock_agent_system import StockAgentSystem
from config import Config

class StockAgentCLI:
    """Command-line interface for the Stock Agent System"""
    
    def __init__(self):
        self.system = None
        self.running = True
    
    async def initialize(self):
        """Initialize the system"""
        print("🚀 Initializing AI Stock Analysis Agent System...")
        print("=" * 50)
        
        try:
            self.system = StockAgentSystem()
            if self.system.is_initialized:
                print("✅ System initialized successfully!")
                await self.show_system_info()
            else:
                print("❌ Failed to initialize system")
                return False
        except Exception as e:
            print(f"❌ Error during initialization: {str(e)}")
            return False
        
        return True
    
    async def show_system_info(self):
        """Display system information"""
        status = self.system.get_system_status()
        capabilities = self.system.get_agent_capabilities()
        tools = self.system.get_available_tools()
        
        print("\n📊 SYSTEM INFORMATION:")
        print(f"Status: {'✅ Initialized' if status['initialized'] else '❌ Not Initialized'}")
        print(f"Model: {status['config']['model']}")
        print(f"Tools Available: {status['tools']}")
        print(f"Session History: {status['session_history_length']} queries")
        
        print("\n🤖 AGENT CAPABILITIES:")
        for agent, caps in capabilities.items():
            print(f"\n{agent.upper()} AGENT:")
            for cap in caps:
                print(f"  • {cap}")
        
        print("\n🛠️ AVAILABLE TOOLS:")
        for tool in tools:
            print(f"  • {tool['name']}: {tool['description']}")
    
    async def run_interactive_mode(self):
        """Run interactive command-line mode"""
        print("\n" + "=" * 50)
        print("🎯 INTERACTIVE MODE")
        print("Type 'help' for commands, 'quit' to exit")
        print("=" * 50)
        
        while self.running:
            try:
                query = input("\n🤖 Enter your query: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    self.running = False
                    break
                
                if query.lower() == 'help':
                    await self.show_help()
                    continue
                
                if query.lower() == 'status':
                    await self.show_system_info()
                    continue
                
                if query.lower() == 'test':
                    await self.test_system()
                    continue
                
                if query.lower() == 'history':
                    await self.show_history()
                    continue
                
                if query.lower() == 'clear':
                    self.system.clear_session_history()
                    print("✅ Session history cleared")
                    continue
                
                # Process the query
                print("\n🔄 Processing query...")
                result = await self.system.analyze_query(query)
                
                if result.get("success"):
                    print("\n📋 RESULT:")
                    print("-" * 30)
                    print(result["output"])
                else:
                    print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                self.running = False
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
    
    async def show_help(self):
        """Show help information"""
        help_text = """
📚 AVAILABLE COMMANDS:

Query Commands:
  • Any stock-related question (e.g., "Get price for AAPL")
  • "Analyze TSLA comprehensively"
  • "Compare AAPL, MSFT, GOOGL"
  • "Portfolio analysis: AAPL, MSFT, GOOGL, TSLA"

System Commands:
  • help     - Show this help message
  • status   - Show system information
  • test     - Test the system
  • history  - Show query history
  • clear    - Clear session history
  • quit     - Exit the application

Example Queries:
  • "What's the current price of Apple stock?"
  • "Get detailed information about Tesla"
  • "Analyze Microsoft stock comprehensively"
  • "Compare the tech giants: AAPL, MSFT, GOOGL"
  • "What's the market outlook for the next 6 months?"
  • "Analyze this portfolio: AAPL, MSFT, GOOGL, TSLA, AMZN"
        """
        print(help_text)
    
    async def test_system(self):
        """Test the system with sample queries"""
        print("\n🧪 TESTING SYSTEM...")
        
        test_queries = [
            "Get the current stock price for AAPL",
            "Get detailed information about MSFT",
            "Analyze TSLA comprehensively"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Test {i}: {query}")
            print("-" * 40)
            
            result = await self.system.analyze_query(query)
            
            if result.get("success"):
                print("✅ SUCCESS")
                print(result["output"][:200] + "..." if len(result["output"]) > 200 else result["output"])
            else:
                print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
            
            # Small delay between tests
            await asyncio.sleep(1)
        
        print("\n✅ System testing completed!")
    
    async def show_history(self):
        """Show query history"""
        history = self.system.get_session_history()
        
        if not history:
            print("📝 No queries in history")
            return
        
        print(f"\n📝 QUERY HISTORY ({len(history)} queries):")
        print("-" * 50)
        
        for i, entry in enumerate(history[-10:], 1):  # Show last 10 queries
            print(f"{i}. Query: {entry['query']}")
            print(f"   Success: {'✅' if entry['result'].get('success') else '❌'}")
            print(f"   Agent: {entry['result'].get('agent', 'Unknown')}")
            print()

async def run_examples():
    """Run example queries to demonstrate the system"""
    print("🎯 RUNNING EXAMPLES...")
    print("=" * 50)
    
    # Initialize system
    system = StockAgentSystem()
    
    if not system.is_initialized:
        print("❌ Failed to initialize system for examples")
        return
    
    # Example 1: Get stock price
    print("\n📊 Example 1: Get Stock Price")
    print("-" * 30)
    result = await system.get_stock_price("AAPL")
    if result.get("success"):
        print("✅ SUCCESS")
        print(result["output"])
    else:
        print(f"❌ FAILED: {result.get('error')}")
    
    # Example 2: Get stock info
    print("\n📊 Example 2: Get Stock Information")
    print("-" * 30)
    result = await system.get_stock_info("MSFT")
    if result.get("success"):
        print("✅ SUCCESS")
        print(result["output"])
    else:
        print(f"❌ FAILED: {result.get('error')}")
    
    # Example 3: Comprehensive analysis
    print("\n📊 Example 3: Comprehensive Analysis")
    print("-" * 30)
    result = await system.analyze_stock("TSLA")
    if result.get("success"):
        print("✅ SUCCESS")
        print(result["output"])
    else:
        print(f"❌ FAILED: {result.get('error')}")
    
    # Example 4: Compare stocks
    print("\n📊 Example 4: Compare Stocks")
    print("-" * 30)
    result = await system.compare_stocks(["AAPL", "MSFT", "GOOGL"])
    if result.get("success"):
        print("✅ SUCCESS")
        print(result["output"])
    else:
        print(f"❌ FAILED: {result.get('error')}")
    
    print("\n✅ Examples completed!")

async def main():
    """Main function"""
    print("🤖 AI Stock Analysis Agent System")
    print("=" * 50)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "examples":
            await run_examples()
            return
        elif sys.argv[1] == "test":
            system = StockAgentSystem()
            if system.is_initialized:
                result = await system.test_system()
                print(f"Test Result: {result}")
            return
    
    # Run interactive mode
    cli = StockAgentCLI()
    if await cli.initialize():
        await cli.run_interactive_mode()
    else:
        print("❌ Failed to start interactive mode")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}") 