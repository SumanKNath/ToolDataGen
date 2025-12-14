"""
Function Generator Module

This module generates Python function implementations from specifications using LLM.
All generated functions operate on a local SQLite database to avoid external side effects.
Generated code is wrapped in an MCP server for easy integration.

Usage:
    generator = FunctionGenerator()
    generator.load_spec_from_json("specs.json")  # or load_spec_from_text() or load_spec_from_url()
    generator.generate_all()
    generator.save_to_directory("output_dir")
"""

import os
import json
import re
import sqlite3
import urllib.request
import subprocess
import sys
import shutil
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from llm_client import LLMCLient

@dataclass
class FunctionSpec:
    """Represents a function specification."""
    name: str
    description: str
    parameters: List[Dict[str, Any]]  # List of {"name": str, "type": str, "description": str}
    returns: Dict[str, Any]  # {"type": str, "description": str}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "returns": self.returns
        }

@dataclass
class GeneratedFunction:
    """Represents a generated function with its code and metadata."""
    spec: FunctionSpec
    code: str
    test_code: str = ""
    db_tables: List[str] = field(default_factory=list)
    passed_tests: bool = False

class FunctionGenerator:
    """
    Generates Python function implementations from specifications using LLM.
    
    All generated functions operate on a local SQLite database to ensure
    no external side effects. Operations like file creation, network messages,
    etc. are simulated by storing data in the database.
    
    Generated output includes:
    - functions.py: Generated function implementations
    - tests.py: Test cases for the functions
    - mcp_server.py: MCP server wrapping all functions
    - function_state.db: SQLite database for state management
    
    Attributes:
        db_path: Path to the SQLite database file
        specs: List of function specifications to generate
        generated_functions: List of generated function implementations
    """
    
    MAX_RETRIES = 3  # Maximum retries for function generation if tests fail
    
    def __init__(self, db_path: str = "function_state.db"):
        """
        Initialize the FunctionGenerator.
        
        Args:
            db_path: Path to the SQLite database file for storing function state
        """
        self.db_path = db_path
        self.specs: List[FunctionSpec] = []
        self.generated_functions: List[GeneratedFunction] = []
        self.llm_client = LLMCLient()
        self.db_schema: str = ""
        self.spec_name: str = "generated"  # Name for the output directory
        
    def load_spec_from_json(self, json_path: str) -> None:
        """
        Load function specifications from a JSON file.
        
        Args:
            json_path: Path to the JSON file containing function specs
            
        Expected JSON format:
        {
            "functions": [
                {
                    "name": "function_name",
                    "description": "What the function does",
                    "parameters": [
                        {"name": "param1", "type": "str", "description": "..."}
                    ],
                    "returns": {"type": "bool", "description": "..."}
                }
            ]
        }
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract spec name from file name
        self.spec_name = os.path.splitext(os.path.basename(json_path))[0]
        self._parse_spec_data(data)
    
    def load_spec_from_text(self, text: str, spec_name: str = "text_spec") -> None:
        """
        Load function specifications from plain text.
        
        Uses LLM to parse the text into structured function specifications.
        
        Args:
            text: Plain text description of functions to generate
            spec_name: Name for the output directory
        """
        self.spec_name = spec_name
        
        system_prompt = """You are a function specification parser. 
        Parse the given text and extract function specifications.
        Return a JSON object with the following structure:
        ```json
        {
            "functions": [
                {
                    "name": "function_name",
                    "description": "What the function does",
                    "parameters": [
                        {"name": "param1", "type": "str", "description": "Parameter description"}
                    ],
                    "returns": {"type": "return_type", "description": "Return value description"}
                }
            ]
        }
        ```
        Use Python type hints for types (str, int, float, bool, List[type], Dict[str, type], Optional[type], etc.)
        """
        
        user_prompt = f"Parse the following function specifications:\n\n{text}"
        
        data = self.llm_client.get_response_json(user_prompt, system_prompt)
        if data:
            self._parse_spec_data(data)
        else:
            raise ValueError("Failed to parse function specifications from text")
    
    def load_spec_from_url(self, url: str) -> None:
        """
        Load function specifications from a URL.
        
        Fetches the content from the URL and uses LLM to parse it.
        
        Args:
            url: URL to fetch function specifications from
        """
        # Extract spec name from URL
        url_parts = url.rstrip('/').split('/')
        self.spec_name = url_parts[-1] if url_parts[-1] else url_parts[-2]
        self.spec_name = re.sub(r'[^a-zA-Z0-9_-]', '_', self.spec_name)
        
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read().decode('utf-8')
        
        # Try to extract text content from HTML
        # Remove script and style tags
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags but keep text
        content = re.sub(r'<[^>]+>', ' ', content)
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Use LLM to parse the web content
        system_prompt = """You are a function specification parser. 
        Parse the given web page content and extract function specifications.
        Return a JSON object with the following structure:
        ```json
        {
            "functions": [
                {
                    "name": "function_name",
                    "description": "What the function does",
                    "parameters": [
                        {"name": "param1", "type": "str", "description": "Parameter description"}
                    ],
                    "returns": {"type": "return_type", "description": "Return value description"}
                }
            ]
        }
        ```
        Use Python type hints for types (str, int, float, bool, List[type], Dict[str, type], Optional[type], etc.)
        Extract ALL function/tool definitions you can find in the content.
        Look for patterns like function names, descriptions, parameters, and return types.
        Make sure to return valid JSON wrapped in ```json code blocks.
        """
        
        # Use more content for better parsing
        user_prompt = f"Parse function specifications from this web page content:\n\n{content[:30000]}"
        
        data = self.llm_client.get_response_json(user_prompt, system_prompt)
        if data:
            self._parse_spec_data(data)
        else:
            # Try with a simpler approach - ask for text response and manually parse
            print("JSON parsing failed, trying text-based approach...")
            text_response = self.llm_client.get_response_text(user_prompt, system_prompt)
            # Try to find JSON in the response
            json_patterns = [
                r'```json\s*(.*?)\s*```',
                r'```\s*(.*?)\s*```',
                r'\{[\s\S]*"functions"[\s\S]*\}'
            ]
            for pattern in json_patterns:
                match = re.search(pattern, text_response, re.DOTALL)
                if match:
                    try:
                        json_text = match.group(1) if '```' in pattern else match.group(0)
                        data = json.loads(json_text)
                        self._parse_spec_data(data)
                        return
                    except json.JSONDecodeError:
                        continue
            raise ValueError("Failed to parse function specifications from URL")
    
    def _parse_spec_data(self, data: Dict[str, Any]) -> None:
        """
        Parse specification data into FunctionSpec objects.
        
        Args:
            data: Dictionary containing function specifications
        """
        functions = data.get("functions", [])
        for func_data in functions:
            spec = FunctionSpec(
                name=func_data.get("name", ""),
                description=func_data.get("description", ""),
                parameters=func_data.get("parameters", []),
                returns=func_data.get("returns", {"type": "None", "description": ""})
            )
            self.specs.append(spec)
    
    def generate_database_schema(self) -> str:
        """
        Generate SQLite database schema based on all function specifications.
        
        Uses LLM to analyze the functions and create appropriate tables
        to support all operations.
        
        Returns:
            SQL schema string for creating the database tables
        """
        specs_json = json.dumps([s.to_dict() for s in self.specs], indent=2)
        
        system_prompt = """You are a database schema designer.
        Given function specifications, design a SQLite database schema that can support all the functions.
        
        IMPORTANT RULES:
        1. All external side effects must be simulated in the database:
           - File operations -> store in a 'files' table with columns: id, path, content, created_at, modified_at
           - Network operations -> store in a 'network_messages' table with columns: id, type, destination, payload, timestamp
           - Email operations -> store in an 'emails' table with columns: id, sender, recipient, subject, body, sent_at
           - Any other external operation -> create appropriate table
        
        2. Include a 'system_state' table for general key-value state storage
        
        3. Each table should have appropriate indexes for common queries
        
        4. Use appropriate SQLite data types (TEXT, INTEGER, REAL, BLOB)
        
        5. Include created_at/updated_at timestamps where appropriate
        
        Return ONLY the SQL CREATE TABLE statements, nothing else.
        Each statement should end with a semicolon.
        Do not include any explanation, just the SQL code.
        """
        
        user_prompt = f"Design a database schema for these functions:\n\n{specs_json}"
        
        schema = self.llm_client.get_response_text(user_prompt, system_prompt)
        
        # Clean up the schema (remove markdown code blocks if present)
        schema = re.sub(r'```sql\s*', '', schema)
        schema = re.sub(r'```\s*', '', schema)
        
        # Always include base tables
        base_tables = """
-- Base system state table for key-value storage
CREATE TABLE IF NOT EXISTS system_state (
    key TEXT PRIMARY KEY,
    value TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Virtual file system table
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    content BLOB,
    is_directory INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Network messages table for simulating network operations
CREATE TABLE IF NOT EXISTS network_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_type TEXT NOT NULL,
    destination TEXT,
    source TEXT,
    payload TEXT,
    headers TEXT,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Email table for simulating email operations
CREATE TABLE IF NOT EXISTS emails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sender TEXT,
    recipient TEXT,
    subject TEXT,
    body TEXT,
    attachments TEXT,
    status TEXT DEFAULT 'draft',
    sent_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

"""
        self.db_schema = base_tables + schema
        return self.db_schema
    
    def generate_function_code(self, spec: FunctionSpec) -> str:
        """
        Generate implementation code for a single function specification.
        
        Args:
            spec: The function specification to implement
            
        Returns:
            Python code string for the function
        """
        system_prompt = """You are a Python code generator.
        Generate a Python function implementation based on the given specification.
        
        CRITICAL RULES:
        1. The function MUST use SQLite database for ALL state management
        2. NO external side effects allowed:
           - Instead of writing files, store in 'files' table
           - Instead of sending network requests, store in 'network_messages' table
           - Instead of sending emails, store in 'emails' table
           - Use 'system_state' table for any other persistent state
        
        3. The function should accept 'db_path' as an optional parameter (default: 'function_state.db')
        
        4. Include proper error handling with try/except blocks
        
        5. Include a comprehensive docstring describing:
           - What the function does
           - Parameters and their types
           - Return value and type
           - Any database tables it uses
        
        6. Use type hints for all parameters and return values
        
        7. The function should be self-contained and importable
        
        8. Do NOT use $ in parameter names - use valid Python identifiers
        
        Return ONLY the Python function code, nothing else.
        Do not include any markdown formatting or code blocks.
        Do not include import statements - they will be added separately.
        """
        
        spec_json = json.dumps(spec.to_dict(), indent=2)
        
        user_prompt = f"""Generate Python code for this function:

{spec_json}

Database schema available:
{self.db_schema}

Remember: ALL side effects must be stored in the SQLite database. No actual file I/O, network calls, etc.
Do NOT use $ in parameter names.
"""
        
        code = self.llm_client.get_response_text(user_prompt, system_prompt)
        
        # Clean up code (remove markdown if present)
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)
        
        # Fix any $ in parameter names
        code = re.sub(r'\$(\w+)', r'\1', code)
        
        return code.strip()
    
    def generate_test_code(self, spec: FunctionSpec, function_code: str) -> str:
        """
        Generate test code for a function.
        
        Args:
            spec: The function specification
            function_code: The generated function code
            
        Returns:
            Python test code string
        """
        system_prompt = """You are a Python test code generator.
        Generate pytest test cases for the given function.
        
        RULES:
        1. Generate at least 3 test cases covering:
           - Normal/happy path
           - Edge cases
           - Error conditions
        
        2. Each test should:
           - Call Reset() first to ensure clean state
           - Use a test database path like 'test_db.db'
           - Assert expected outcomes
        
        3. Test function names should start with 'test_'
        
        4. Include docstrings for each test explaining what it tests
        
        5. Do NOT include import statements - they will be added separately
        
        Return ONLY the test functions, nothing else.
        Do not include any markdown formatting or code blocks.
        """
        
        spec_json = json.dumps(spec.to_dict(), indent=2)
        
        user_prompt = f"""Generate pytest test cases for this function:

Specification:
{spec_json}

Function code:
{function_code}

Generate tests that verify the function works correctly with the SQLite database.
"""
        
        test_code = self.llm_client.get_response_text(user_prompt, system_prompt)
        
        # Clean up code
        test_code = re.sub(r'```python\s*', '', test_code)
        test_code = re.sub(r'```\s*', '', test_code)
        
        return test_code.strip()
    
    def generate_reset_function(self) -> str:
        """
        Generate a Reset() function that initializes the database with deterministic state.
        
        Returns:
            Python code string for the Reset function
        """
        system_prompt = """You are a Python code generator.
        Generate a Reset() function that:
        1. Creates/recreates the SQLite database with the given schema
        2. Populates it with deterministic initial data
        3. The function should accept 'db_path' as an optional parameter (default: 'function_state.db')
        
        Return ONLY the Python function code, nothing else.
        Do not include any markdown formatting or code blocks.
        Do not include import statements.
        Include a comprehensive docstring.
        """
        
        specs_summary = "\n".join([f"- {s.name}: {s.description}" for s in self.specs])
        
        user_prompt = f"""Generate a Reset() function for a database with this schema:

{self.db_schema}

The database supports these functions:
{specs_summary}

The Reset function should:
1. Drop all existing tables
2. Recreate the schema
3. Insert some sample/initial data that would be useful for testing the functions
4. Use deterministic data (no random values) so results are reproducible
5. Do NOT use single quotes inside string literals that are already quoted with single quotes
"""
        
        code = self.llm_client.get_response_text(user_prompt, system_prompt)
        
        # Clean up code
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)
        
        return code.strip()
    
    def _sanitize_param_name(self, name: str) -> str:
        """
        Sanitize parameter name to be a valid Python identifier.
        
        Args:
            name: Original parameter name
            
        Returns:
            Sanitized parameter name
        """
        # Replace hyphens with underscores
        sanitized = name.replace("-", "_")
        # Replace dots with underscores
        sanitized = sanitized.replace(".", "_")
        # Remove @ symbols
        sanitized = sanitized.replace("@", "")
        # Remove $ symbols
        sanitized = sanitized.replace("$", "")
        # Remove any other non-alphanumeric characters except underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '', sanitized)
        # Ensure it starts with a letter or underscore
        if sanitized and sanitized[0].isdigit():
            sanitized = "_" + sanitized
        # Handle empty string
        if not sanitized:
            sanitized = "param"
        return sanitized
    
    def generate_mcp_server(self, function_names: List[str]) -> str:
        """
        Generate MCP server code using FastMCP framework.
        
        FastMCP handles parameter mapping automatically using function introspection,
        which ensures parameter names match correctly between the MCP interface
        and the actual function signatures.
        
        Args:
            function_names: List of function names to expose
            
        Returns:
            Python code string for the MCP server
        """
        function_imports = ", ".join(function_names + ["Reset"])
        
        # Generate tool wrapper functions that use **kwargs for robust parameter handling
        tool_wrappers = []
        for spec in self.specs:
            # Get the actual parameter names from the spec (sanitized)
            params_with_types = []
            param_docs = []
            for p in spec.parameters:
                if p["name"] != "db_path":
                    sanitized = self._sanitize_param_name(p["name"])
                    ptype = p.get("type", "str")
                    # Convert Python type hints to simple types for FastMCP
                    if "List" in ptype:
                        ptype = "list"
                    elif "Dict" in ptype:
                        ptype = "dict"
                    elif "Optional" in ptype:
                        # Extract inner type
                        inner = ptype.replace("Optional[", "").rstrip("]")
                        ptype = inner if inner else "str"
                    params_with_types.append(f'{sanitized}: {ptype} = None')
                    param_docs.append(f'        {sanitized}: {p.get("description", "")}')
            
            params_str = ", ".join(params_with_types)
            param_docs_str = "\n".join(param_docs) if param_docs else "        None"
            
            # Build kwargs dict to pass to actual function
            kwargs_items = []
            for p in spec.parameters:
                if p["name"] != "db_path":
                    sanitized = self._sanitize_param_name(p["name"])
                    kwargs_items.append(f'"{sanitized}": {sanitized}')
            kwargs_str = ", ".join(kwargs_items)
            
            wrapper = f'''
@mcp.tool()
def {spec.name}({params_str}) -> str:
    """
    {spec.description}
    
    Args:
{param_docs_str}
    
    Returns:
        Result of the operation as JSON string
    """
    kwargs = {{{kwargs_str}}}
    # Filter out None values
    kwargs = {{k: v for k, v in kwargs.items() if v is not None}}
    result = _funcs.{spec.name}(**kwargs, db_path=DB_PATH)
    return json.dumps({{"result": result}})
'''
            tool_wrappers.append(wrapper)
        
        tool_wrappers_str = "\n".join(tool_wrappers)
        
        mcp_server_code = f'''"""
MCP Server for Generated Functions

This module provides an MCP (Model Context Protocol) server using FastMCP framework
that exposes the generated functions as tools that can be called by AI assistants.

FastMCP handles parameter mapping automatically via function introspection.

Requirements:
    pip install fastmcp

Usage:
    python mcp_server.py
    
    Or with uvx:
    uvx fastmcp run mcp_server.py
"""

import json
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastmcp import FastMCP

# Import all generated functions
import functions as _funcs
from functions import Reset

# Database path for the MCP server
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "function_state.db")

# Create FastMCP server
mcp = FastMCP("Generated Functions Server")

# Initialize database on module load
Reset(db_path=DB_PATH)

@mcp.tool()
def reset_database() -> str:
    """
    Reset the database to its initial state with deterministic data.
    
    Returns:
        Status message
    """
    Reset(db_path=DB_PATH)
    return json.dumps({{"status": "success", "message": "Database reset complete"}})

{tool_wrappers_str}

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
'''
        return mcp_server_code
    
    def run_tests(self, output_dir: str) -> Tuple[bool, str]:
        """
        Run the generated tests.
        
        Args:
            output_dir: Directory containing the generated code
            
        Returns:
            Tuple of (success, output)
        """
        tests_path = os.path.join(output_dir, "tests.py")
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", tests_path, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=output_dir
            )
            
            output = result.stdout + result.stderr
            success = result.returncode == 0
            
            return success, output
        except subprocess.TimeoutExpired:
            return False, "Tests timed out"
        except Exception as e:
            return False, str(e)
    
    def _run_single_function_tests(self, output_dir: str, func_name: str) -> Tuple[bool, str]:
        """
        Run tests for a single function.
        
        Args:
            output_dir: Directory containing the generated code
            func_name: Name of the function to test
            
        Returns:
            Tuple of (success, output)
        """
        tests_path = os.path.join(output_dir, "tests.py")
        
        try:
            # Run pytest with -k flag to select tests matching the function name
            result = subprocess.run(
                [sys.executable, "-m", "pytest", tests_path, "-v", "--tb=short", "-k", func_name],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=output_dir
            )
            
            output = result.stdout + result.stderr
            success = result.returncode == 0
            
            return success, output
        except subprocess.TimeoutExpired:
            return False, "Tests timed out"
        except Exception as e:
            return False, str(e)
    
    def _save_incremental_files(self, output_dir: str, func_idx: int) -> None:
        """
        Save functions.py and tests.py with current generated functions for incremental testing.
        
        Args:
            output_dir: Directory to save to
            func_idx: Index of the current function being tested (includes 0..func_idx)
        """
        # Build functions.py content
        functions_header = '''"""
Auto-generated function implementations.

This module contains functions generated from specifications.
All functions operate on a local SQLite database to avoid external side effects.

Generated by FunctionGenerator
"""

import sqlite3
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Database path constant
DEFAULT_DB_PATH = "function_state.db"

def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Get a database connection with row factory enabled.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        SQLite connection object
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

'''
        
        # Add the database schema as a constant
        schema_const = f'''# Database schema
DB_SCHEMA = """
{self.db_schema}
"""

'''
        
        # Combine all parts for functions.py
        functions_content = functions_header + schema_const + self.reset_function_code + "\n\n\n"
        
        for i, gen_func in enumerate(self.generated_functions):
            if i <= func_idx:
                functions_content += gen_func.code + "\n\n\n"
        
        # Write functions.py
        functions_path = os.path.join(output_dir, "functions.py")
        with open(functions_path, 'w', encoding='utf-8') as f:
            f.write(functions_content)
        
        # Build tests.py content
        function_names = [gf.spec.name for i, gf in enumerate(self.generated_functions) if i <= func_idx]
        imports_str = ", ".join(function_names + ["Reset"])
        
        tests_header = f'''"""
Auto-generated tests for function implementations.

Generated by FunctionGenerator
"""

import pytest
import os
import sys
import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from functions import {imports_str}

# Test database path
TEST_DB_PATH = "test_function_state.db"

@pytest.fixture(autouse=True)
def setup_database():
    """Reset database before each test."""
    Reset(db_path=TEST_DB_PATH)
    yield
    # Cleanup after test
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except:
            pass

'''
        
        tests_content = tests_header
        for i, gen_func in enumerate(self.generated_functions):
            if i <= func_idx:
                tests_content += f"\n# Tests for {gen_func.spec.name}\n"
                tests_content += gen_func.test_code + "\n\n"
        
        # Write tests.py
        tests_path = os.path.join(output_dir, "tests.py")
        with open(tests_path, 'w', encoding='utf-8') as f:
            f.write(tests_content)
    
    def generate_all(self) -> None:
        """
        Generate database schema and all function implementations with tests.
        
        This method should be called after loading specifications.
        It will generate the database schema and then generate code
        for each function specification, including tests.
        Each function is tested immediately after generation and 
        regenerated up to MAX_RETRIES times if tests fail.
        """
        if not self.specs:
            raise ValueError("No specifications loaded. Call load_spec_from_* first.")
        
        # Generate database schema
        print("Generating database schema...")
        self.generate_database_schema()
        
        # Generate Reset function
        print("Generating Reset function...")
        self.reset_function_code = self.generate_reset_function()
        
        # Create temp output directory for incremental testing
        temp_output_dir = os.path.join(os.getcwd(), f"_temp_generated_{self.spec_name}")
        os.makedirs(temp_output_dir, exist_ok=True)
        
        # Generate each function with tests and test immediately
        self.generated_functions = []
        for spec_idx, spec in enumerate(self.specs):
            print(f"\n{'='*60}")
            print(f"Generating function {spec_idx + 1}/{len(self.specs)}: {spec.name}")
            print('='*60)
            
            function_passed = False
            
            for attempt in range(self.MAX_RETRIES):
                print(f"\n  Attempt {attempt + 1}/{self.MAX_RETRIES}...")
                
                # Generate function code
                print(f"    Generating implementation...")
                function_code = self.generate_function_code(spec)
                
                # Generate test code
                print(f"    Generating tests...")
                test_code = self.generate_test_code(spec, function_code)
                
                # Create or update the generated function
                if attempt == 0:
                    generated = GeneratedFunction(
                        spec=spec,
                        code=function_code,
                        test_code=test_code
                    )
                    self.generated_functions.append(generated)
                else:
                    # Update existing
                    self.generated_functions[-1].code = function_code
                    self.generated_functions[-1].test_code = test_code
                
                # Save incremental files for testing
                self._save_incremental_files(temp_output_dir, len(self.generated_functions) - 1)
                
                # Initialize database
                db_path = os.path.join(temp_output_dir, "function_state.db")
                self._init_db(db_path)
                
                # Run tests for this specific function
                print(f"    Running tests for {spec.name}...")
                success, output = self._run_single_function_tests(temp_output_dir, spec.name)
                
                if success:
                    print(f"    ✓ Tests passed for {spec.name}")
                    self.generated_functions[-1].passed_tests = True
                    function_passed = True
                    break
                else:
                    # Parse and show failure details
                    print(f"    ✗ Tests failed for {spec.name}")
                    
                    failure_details = self._parse_failure_details(output)
                    errors = failure_details.get(spec.name, [])
                    
                    if errors:
                        print(f"    Failure reasons:")
                        for error in errors[:3]:
                            error_short = error[:150] + "..." if len(error) > 150 else error
                            print(f"      - {error_short}")
                    else:
                        # Show raw output if we couldn't parse specific errors
                        print(f"    Test output (truncated):")
                        # Look for error lines in output
                        error_lines = [line for line in output.split('\n') if 'Error' in line or 'FAILED' in line]
                        for line in error_lines[:5]:
                            print(f"      {line[:100]}")
                    
                    if attempt < self.MAX_RETRIES - 1:
                        print(f"\n    Regenerating {spec.name}...")
            
            if not function_passed:
                print(f"\n  ⚠ Warning: {spec.name} still failing after {self.MAX_RETRIES} attempts")
                print(f"    Continuing with current implementation...")
        
        # Clean up temp directory
        try:
            shutil.rmtree(temp_output_dir)
        except:
            pass
        
        print(f"\n{'='*60}")
        print("Generation complete!")
        print('='*60)
        passed = sum(1 for gf in self.generated_functions if gf.passed_tests)
        print(f"Functions passed: {passed}/{len(self.generated_functions)}")
    
    def _init_db(self, db_path: str) -> None:
        """
        Initialize the SQLite database.
        
        Args:
            db_path: Path to the database file
        """
        # Remove existing db
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except:
                pass
        
        conn = sqlite3.connect(db_path)
        try:
            # Execute schema
            statements = self.db_schema.split(';')
            for stmt in statements:
                stmt = stmt.strip()
                if stmt and not stmt.startswith('--'):
                    if 'CREATE TABLE' in stmt.upper() and 'IF NOT EXISTS' not in stmt.upper():
                        stmt = stmt.replace('CREATE TABLE', 'CREATE TABLE IF NOT EXISTS', 1)
                    try:
                        conn.execute(stmt)
                    except sqlite3.OperationalError:
                        pass  # Ignore errors for incremental building
            conn.commit()
        finally:
            conn.close()
    
    def save_to_directory(self, base_dir: str = None) -> str:
        """
        Save all generated code to a dedicated directory.
        
        Args:
            base_dir: Base directory for output. If None, uses current directory.
            
        Returns:
            Path to the output directory
        """
        if not self.generated_functions:
            raise ValueError("No functions generated. Call generate_all() first.")
        
        # Create output directory
        if base_dir is None:
            base_dir = os.getcwd()
        
        output_dir = os.path.join(base_dir, f"generated_{self.spec_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Build functions.py content
        functions_header = '''"""
Auto-generated function implementations.

This module contains functions generated from specifications.
All functions operate on a local SQLite database to avoid external side effects.

Generated by FunctionGenerator
"""

import sqlite3
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Database path constant
DEFAULT_DB_PATH = "function_state.db"

def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Get a database connection with row factory enabled.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        SQLite connection object
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

'''
        
        # Add the database schema as a constant
        schema_const = f'''# Database schema
DB_SCHEMA = """
{self.db_schema}
"""

'''
        
        # Combine all parts for functions.py
        functions_content = functions_header + schema_const + self.reset_function_code + "\n\n\n"
        
        for gen_func in self.generated_functions:
            functions_content += gen_func.code + "\n\n\n"
        
        # Write functions.py
        functions_path = os.path.join(output_dir, "functions.py")
        with open(functions_path, 'w', encoding='utf-8') as f:
            f.write(functions_content)
        
        # Build tests.py content
        function_names = [gf.spec.name for gf in self.generated_functions]
        imports_str = ", ".join(function_names + ["Reset"])
        
        tests_header = f'''"""
Auto-generated tests for function implementations.

Generated by FunctionGenerator
"""

import pytest
import os
import sys
import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from functions import {imports_str}

# Test database path
TEST_DB_PATH = "test_function_state.db"

@pytest.fixture(autouse=True)
def setup_database():
    """Reset database before each test."""
    Reset(db_path=TEST_DB_PATH)
    yield
    # Cleanup after test
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except:
            pass

'''
        
        tests_content = tests_header
        for gen_func in self.generated_functions:
            tests_content += f"\n# Tests for {gen_func.spec.name}\n"
            tests_content += gen_func.test_code + "\n\n"
        
        # Write tests.py
        tests_path = os.path.join(output_dir, "tests.py")
        with open(tests_path, 'w', encoding='utf-8') as f:
            f.write(tests_content)
        
        # Generate and write MCP server
        function_names = [gf.spec.name for gf in self.generated_functions]
        mcp_server_code = self.generate_mcp_server(function_names)
        
        mcp_path = os.path.join(output_dir, "mcp_server.py")
        with open(mcp_path, 'w', encoding='utf-8') as f:
            f.write(mcp_server_code)
        
        # Save spec as JSON for reference
        spec_data = {
            "functions": [s.to_dict() for s in self.specs]
        }
        spec_path = os.path.join(output_dir, "spec.json")
        with open(spec_path, 'w', encoding='utf-8') as f:
            json.dump(spec_data, f, indent=2)
        
        # Generate requirements.txt
        requirements_content = """# Requirements for generated MCP server
fastmcp>=0.1.0
pytest>=7.0.0
"""
        requirements_path = os.path.join(output_dir, "requirements.txt")
        with open(requirements_path, 'w', encoding='utf-8') as f:
            f.write(requirements_content)
        
        # Generate README.md
        readme_content = f"""# Generated Functions - {self.spec_name}

This directory contains auto-generated function implementations with an MCP server.

## Files

- `functions.py` - Generated function implementations
- `tests.py` - Pytest test cases
- `mcp_server.py` - FastMCP server exposing functions as tools
- `function_state.db` - SQLite database for state management
- `spec.json` - Original function specifications
- `requirements.txt` - Python dependencies

## Setup

```bash
pip install -r requirements.txt
```

## Running the MCP Server

```bash
python mcp_server.py
```

Or using FastMCP CLI:

```bash
fastmcp run mcp_server.py
```

## Running Tests

```bash
pytest tests.py -v
```

## Functions

{chr(10).join([f"- **{s.name}**: {s.description}" for s in self.specs])}

## Database

All functions operate on a local SQLite database (`function_state.db`).
External side effects (file I/O, network calls, emails) are simulated in the database.

Use the `reset_database()` tool to reset the database to its initial state.
"""
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"Generated code saved to: {output_dir}")
        print(f"  - functions.py: Function implementations")
        print(f"  - tests.py: Test cases")
        print(f"  - mcp_server.py: MCP server (FastMCP)")
        print(f"  - spec.json: Function specifications")
        print(f"  - requirements.txt: Python dependencies")
        print(f"  - README.md: Documentation")
        
        return output_dir
    
   
    def _parse_failure_details(self, test_output: str) -> Dict[str, List[str]]:
        """
        Parse test output to extract detailed failure information.
        
        Args:
            test_output: Output from pytest
            
        Returns:
            Dictionary mapping function names to list of error messages
        """
        failures = {}
        
        # Look for FAILED lines with test names
        failed_tests = re.findall(r'FAILED\s+tests\.py::(\w+)', test_output)
        
        # Look for error messages - patterns like "AssertionError:", "TypeError:", etc.
        error_patterns = [
            r'(AssertionError:.*?)(?:\n|$)',
            r'(TypeError:.*?)(?:\n|$)',
            r'(ValueError:.*?)(?:\n|$)',
            r'(AttributeError:.*?)(?:\n|$)',
            r'(NameError:.*?)(?:\n|$)',
            r'(sqlite3\.\w+Error:.*?)(?:\n|$)',
            r'(KeyError:.*?)(?:\n|$)',
            r'(IndexError:.*?)(?:\n|$)',
            r'E\s+(.*?)(?:\n|$)',
        ]
        
        all_errors = []
        for pattern in error_patterns:
            matches = re.findall(pattern, test_output)
            all_errors.extend(matches)
        
        # Try to associate errors with function names
        for test_name in failed_tests:
            # Extract function name from test name (e.g., test_create_user_success -> create_user)
            func_matches = []
            for spec in self.specs:
                if spec.name.lower() in test_name.lower() or test_name.lower().replace('test_', '') in spec.name.lower():
                    func_matches.append(spec.name)
            
            # Find relevant errors in the output near this test
            test_section = test_output
            start_idx = test_output.find(test_name)
            if start_idx != -1:
                end_idx = min(start_idx + 1000, len(test_output))
                test_section = test_output[start_idx:end_idx]
            
            section_errors = []
            for pattern in error_patterns:
                matches = re.findall(pattern, test_section)
                section_errors.extend(matches)
            
            for func_name in func_matches:
                if func_name not in failures:
                    failures[func_name] = []
                failures[func_name].extend(section_errors[:3])  # Limit errors per function
        
        # If no specific matches, add all errors to all failed functions
        if not failures and failed_tests:
            for test_name in failed_tests:
                for spec in self.specs:
                    if spec.name not in failures:
                        failures[spec.name] = all_errors[:3]
        
        return failures
    
    def _parse_failing_tests(self, test_output: str) -> List[str]:
        """
        Parse test output to find failing function names.
        
        Args:
            test_output: Output from pytest
            
        Returns:
            List of function names that have failing tests
        """
        failing = set()
        
        # Look for patterns like "FAILED tests.py::test_function_name"
        for match in re.finditer(r'FAILED.*?test_(\w+)', test_output):
            func_name = match.group(1)
            # Try to match to actual function names
            for spec in self.specs:
                if spec.name.lower() in func_name.lower() or func_name.lower() in spec.name.lower():
                    failing.add(spec.name)
        
        # If no specific failures found, regenerate all
        if not failing:
            failing = {spec.name for spec in self.specs}
        
        return list(failing)
    
    def initialize_database(self, output_dir: str) -> None:
        """
        Initialize the SQLite database in the output directory.
        
        Args:
            output_dir: Directory containing the generated code
        """
        db_path = os.path.join(output_dir, "function_state.db")
        
        conn = sqlite3.connect(db_path)
        try:
            # Execute schema
            statements = self.db_schema.split(';')
            for stmt in statements:
                stmt = stmt.strip()
                if stmt and not stmt.startswith('--'):
                    if 'CREATE TABLE' in stmt.upper() and 'IF NOT EXISTS' not in stmt.upper():
                        stmt = stmt.replace('CREATE TABLE', 'CREATE TABLE IF NOT EXISTS', 1)
                    try:
                        conn.execute(stmt)
                    except sqlite3.OperationalError as e:
                        print(f"Warning: {e}")
            conn.commit()
            print(f"Database initialized: {db_path}")
        finally:
            conn.close()

def main():
    """Example usage of FunctionGenerator."""    
   
    # Generate functions
    generator = FunctionGenerator()
    #generator.load_spec_from_json(spec_path)
    generator.load_spec_from_url("https://learn.microsoft.com/en-us/microsoft-agent-365/mcp-server-reference/teams")
    generator.generate_all()
    output_dir = generator.save_to_directory()
    generator.initialize_database(output_dir)
    
    print("\nGeneration complete!")
    print(f"Output directory: {output_dir}")
    print("\nTo run the MCP server:")
    print(f"  cd {output_dir}")
    print("  python mcp_server.py")

if __name__ == "__main__":
    main()
