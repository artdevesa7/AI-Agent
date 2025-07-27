import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the AI Agent system"""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    
    # Stock API Configuration
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
    
    # Agent Configuration
    MASTER_AGENT_TEMPERATURE = float(os.getenv("MASTER_AGENT_TEMPERATURE", "0.7"))
    JUNIOR_AGENT_TEMPERATURE = float(os.getenv("JUNIOR_AGENT_TEMPERATURE", "0.5"))
    ORCHESTRATOR_AGENT_TEMPERATURE = float(os.getenv("ORCHESTRATOR_AGENT_TEMPERATURE", "0.3"))
    
    # System Configuration
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
    VERBOSE = os.getenv("VERBOSE", "True").lower() == "true"
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Please set it in your .env file")
        
        if not cls.ALPHA_VANTAGE_API_KEY and not cls.FINNHUB_API_KEY:
            print("Warning: No stock API key found. Using yfinance as fallback.")
    
    @classmethod
    def get_stock_api_key(cls):
        """Get the preferred stock API key"""
        return cls.ALPHA_VANTAGE_API_KEY or cls.FINNHUB_API_KEY 