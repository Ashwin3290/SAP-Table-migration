"""
Central configuration for TableLLM
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LLM Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GENERATION_MODEL = "gemini-2.0-flash-thinking-exp-01-21"
PLANNING_MODEL = "gemini-2.0-flash"
TEMPERATURE = 0.2
TOP_P = 0.95
TOP_K = 40

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspaces")
GENERATED_CODE_DIR = os.path.join(BASE_DIR, "generated_code")

# Create directories if they don't exist
for directory in [SESSIONS_DIR, WORKSPACE_DIR, GENERATED_CODE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Database Configuration
DATABASE_PATH = os.path.join(BASE_DIR, "db.sqlite3")

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
TOKEN_LOG_PATH = os.path.join(BASE_DIR, "token_usage.log")
AGENT_LOG_PATH = os.path.join(BASE_DIR, "agent_usage.log")

# Agent Configuration
MAX_RETRY_ATTEMPTS = 3
VALIDATION_TIMEOUT = 30  # seconds

# SAP-specific Configuration
SAP_FIELD_LENGTH = {
    "MATNR": 18,  # Material Number
    "MTART": 4,   # Material Type
    "WERKS": 4,   # Plant
    "LGORT": 4,   # Storage Location
    "KUNNR": 10,  # Customer Number
    "LIFNR": 10,  # Vendor Number
    "VBELN": 10,  # Sales Document
    "EBELN": 10,  # Purchasing Document
}

# Workspace Configuration
DEFAULT_WORKSPACES = {
    "Material_Management": {
        "tables": ["MARA", "MARA_500", "MARA_800", "MARC", "MAKT"],
        "description": "Material master data and related tables",
    },
    "Customer_Management": {
        "tables": ["KNA1", "KNB1", "KNVV", "KNVI"],
        "description": "Customer master data and related tables",
    },
    "Reference_Tables": {
        "tables": ["T001W", "T006"],
        "description": "Validation and reference tables",
    }
}
