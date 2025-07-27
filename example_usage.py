#!/usr/bin/env python3
"""
Example usage of the AI Stock Analysis Agent System
Demonstrates various use cases and capabilities
"""

import asyncio
import json
from typing import List, Dict, Any
from stock_agent_system import StockAgentSystem

async def basic_examples():
    """Basic usage examples"""
    print("🚀 Basic Examples")
    print("=" * 50)
    
    # Initialize the system
    system = StockAgentSystem()
    
    if not system.is_initialized:
        print("❌ System failed to initialize")
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
    
    # Example 2: Get stock information
    print("\n📊 Example 2: Get Stock Information")
    print("-" * 30)
    result = await system.get_stock_info("MSFT")
    if result.get("success"):
        print("✅ SUCCESS")
        print(result["output"])
    else:
        print(f"❌ FAILED: {result.get('error')}")
    
    # Example 3: Basic analysis
    print("\n📊 Example 3: Basic Analysis")
    print("-" * 30)
    result = await system.analyze_query("Analyze TSLA stock")
    if result.get("success"):
        print("✅ SUCCESS")
        print(result["output"])
    else:
        print(f"❌ FAILED: {result.get('error')}")

async def advanced_examples():
    """Advanced usage examples"""
    print("\n🎯 Advanced Examples")
    print("=" * 50)
    
    system = StockAgentSystem()
    
    if not system.is_initialized:
        print("❌ System failed to initialize")
        return
    
    # Example 1: Comprehensive analysis
    print("\n📊 Example 1: Comprehensive Analysis")
    print("-" * 30)
    result = await system.analyze_stock("TSLA")
    if result.get("success"):
        print("✅ SUCCESS")
        print(result["output"])
    else:
        print(f"❌ FAILED: {result.get('error')}")
    
    # Example 2: Compare multiple stocks
    print("\n📊 Example 2: Compare Stocks")
    print("-" * 30)
    result = await system.compare_stocks(["AAPL", "MSFT", "GOOGL"])
    if result.get("success"):
        print("✅ SUCCESS")
        print(result["output"])
    else:
        print(f"❌ FAILED: {result.get('error')}")
    
    # Example 3: Portfolio analysis
    print("\n📊 Example 3: Portfolio Analysis")
    print("-" * 30)
    portfolio = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    result = await system.portfolio_analysis(portfolio)
    if result.get("success"):
        print("✅ SUCCESS")
        print(result["output"])
    else:
        print(f"❌ FAILED: {result.get('error')}")

async def custom_queries():
    """Custom query examples"""
    print("\n🎨 Custom Query Examples")
    print("=" * 50)
    
    system = StockAgentSystem()
    
    if not system.is_initialized:
        print("❌ System failed to initialize")
        return
    
    # Custom queries
    custom_queries = [
        "What's the market outlook for electric vehicle stocks?",
        "Compare the financial health of AAPL and MSFT",
        "Analyze the risk factors for investing in TSLA",
        "What are the key metrics to watch for GOOGL?",
        "Provide a sector analysis for technology stocks"
    ]
    
    for i, query in enumerate(custom_queries, 1):
        print(f"\n📝 Custom Query {i}: {query}")
        print("-" * 50)
        
        result = await system.analyze_query(query)
        
        if result.get("success"):
            print("✅ SUCCESS")
            print(result["output"][:300] + "..." if len(result["output"]) > 300 else result["output"])
        else:
            print(f"❌ FAILED: {result.get('error')}")
        
        # Small delay between queries
        await asyncio.sleep(1)

async def system_information():
    """Display system information"""
    print("\n🔧 System Information")
    print("=" * 50)
    
    system = StockAgentSystem()
    
    if not system.is_initialized:
        print("❌ System failed to initialize")
        return
    
    # Get system status
    status = system.get_system_status()
    print(f"System Initialized: {status['initialized']}")
    print(f"Model: {status['config']['model']}")
    print(f"Tools Available: {status['tools']}")
    print(f"Session History: {status['session_history_length']} queries")
    
    # Get agent capabilities
    capabilities = system.get_agent_capabilities()
    print(f"\nAgent Capabilities:")
    for agent, caps in capabilities.items():
        print(f"\n{agent.upper()} AGENT:")
        for cap in caps:
            print(f"  • {cap}")
    
    # Get available tools
    tools = system.get_available_tools()
    print(f"\nAvailable Tools:")
    for tool in tools:
        print(f"  • {tool['name']}: {tool['description']}")

async def session_management():
    """Demonstrate session management"""
    print("\n📝 Session Management")
    print("=" * 50)
    
    system = StockAgentSystem()
    
    if not system.is_initialized:
        print("❌ System failed to initialize")
        return
    
    # Make some queries
    queries = [
        "Get the current price of AAPL",
        "Analyze MSFT stock",
        "Compare GOOGL and TSLA"
    ]
    
    for query in queries:
        print(f"\n🔄 Processing: {query}")
        result = await system.analyze_query(query)
        print(f"Success: {result.get('success')}")
    
    # Show session history
    history = system.get_session_history()
    print(f"\n📋 Session History ({len(history)} queries):")
    for i, entry in enumerate(history, 1):
        print(f"{i}. {entry['query']} - {'✅' if entry['result'].get('success') else '❌'}")
    
    # Clear history
    system.clear_session_history()
    print(f"\n🧹 Session history cleared")

async def error_handling():
    """Demonstrate error handling"""
    print("\n⚠️ Error Handling Examples")
    print("=" * 50)
    
    system = StockAgentSystem()
    
    if not system.is_initialized:
        print("❌ System failed to initialize")
        return
    
    # Test with invalid stock symbol
    print("\n📊 Testing with invalid stock symbol")
    print("-" * 30)
    result = await system.get_stock_price("INVALID_SYMBOL_12345")
    print(f"Result: {result}")
    
    # Test with empty query
    print("\n📊 Testing with empty query")
    print("-" * 30)
    result = await system.analyze_query("")
    print(f"Result: {result}")
    
    # Test with very complex query
    print("\n📊 Testing with complex query")
    print("-" * 30)
    complex_query = "Analyze the macroeconomic factors affecting the technology sector, including interest rates, inflation, and geopolitical risks, while considering the impact on major tech companies like AAPL, MSFT, GOOGL, and TSLA, and provide a comprehensive investment strategy for the next 12 months"
    result = await system.analyze_query(complex_query)
    print(f"Success: {result.get('success')}")
    if result.get("success"):
        print("Complex query handled successfully")

async def performance_test():
    """Test system performance"""
    print("\n⚡ Performance Test")
    print("=" * 50)
    
    system = StockAgentSystem()
    
    if not system.is_initialized:
        print("❌ System failed to initialize")
        return
    
    import time
    
    # Test multiple queries
    test_queries = [
        "Get price for AAPL",
        "Get price for MSFT", 
        "Get price for GOOGL",
        "Get price for TSLA",
        "Get price for AMZN"
    ]
    
    start_time = time.time()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔄 Query {i}: {query}")
        result = await system.analyze_query(query)
        print(f"Success: {result.get('success')}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n⏱️ Performance Results:")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Time per Query: {total_time/len(test_queries):.2f} seconds")
    print(f"Queries per Second: {len(test_queries)/total_time:.2f}")

async def main():
    """Run all examples"""
    print("🤖 AI Stock Analysis Agent System - Examples")
    print("=" * 60)
    
    # Run all example functions
    await basic_examples()
    await advanced_examples()
    await custom_queries()
    await system_information()
    await session_management()
    await error_handling()
    await performance_test()
    
    print("\n✅ All examples completed!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Examples interrupted by user")
    except Exception as e:
        print(f"❌ Error running examples: {str(e)}") 