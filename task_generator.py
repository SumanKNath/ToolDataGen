"""
Task Generator Module

This module generates unique tasks using two different algorithms:
1. Tools-First: Generate tool sequence first, then task description
2. Task-First: Generate task description first, then identify tools

The module uses an OOP structure with a base class containing common functionality
and derived classes implementing specific algorithms.

Usage:
    generator = TaskGenerator("generated_dir")
    generator.generate_tasks(num_tasks=50, output_dir="tasks")
"""

from abc import ABC, abstractmethod
import os
import json
import re
import sqlite3
import sys
import importlib.util
import copy
import shutil
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from llm_client import LLMCLient

# =============================================================================
# SQL QUERY LOGGING AND PARSING
# =============================================================================

class SQLQueryLogger:
    """
    Uses SQLite's built-in set_trace_callback to log all executed SQL queries.
    This is a cleaner approach than monkey-patching sqlite3.connect.
    
    Handles transaction rollbacks by buffering queries during transactions
    and discarding them if a ROLLBACK occurs.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.queries: List[str] = []  # List of committed SQL statements
        self._is_logging = False
        self._transaction_buffer: List[str] = []  # Buffer for current transaction
        self._in_transaction = False
    
    def _trace_callback(self, statement: str):
        """
        Callback function called by SQLite for each executed statement.
        
        Handles transactions:
        - BEGIN: Start buffering queries
        - COMMIT: Flush buffer to committed queries
        - ROLLBACK: Discard buffered queries
        """
        if not self._is_logging:
            return
        
        stmt_upper = statement.strip().upper()
        
        # Check for transaction control statements
        if stmt_upper.startswith('BEGIN'):
            # Start a new transaction - begin buffering
            self._in_transaction = True
            self._transaction_buffer = []
            return
        
        if stmt_upper.startswith('COMMIT'):
            # Transaction committed - flush buffer to committed queries
            if self._in_transaction:
                self.queries.extend(self._transaction_buffer)
                self._transaction_buffer = []
                self._in_transaction = False
            return
        
        if stmt_upper.startswith('ROLLBACK'):
            # Transaction rolled back - discard buffered queries
            if self._in_transaction:
                self._transaction_buffer = []
                self._in_transaction = False
            return
        
        # Regular statement - add to appropriate list
        if self._in_transaction:
            self._transaction_buffer.append(statement)
        else:
            # Not in explicit transaction (autocommit mode)
            self.queries.append(statement)
    
    def start_logging(self, conn: sqlite3.Connection = None):
        """
        Start logging SQL queries using SQLite's trace callback.
        
        If a connection is provided, sets the trace callback on it.
        Otherwise, stores the callback to be set on connections later.
        """
        self.queries = []
        self._is_logging = True
        if conn:
            conn.set_trace_callback(self._trace_callback)
    
    def stop_logging(self, conn: sqlite3.Connection = None):
        """Stop logging and optionally clear the trace callback."""
        self._is_logging = False
        if conn:
            conn.set_trace_callback(None)
    
    def setup_connection(self, conn: sqlite3.Connection):
        """Set up trace callback on a connection."""
        conn.set_trace_callback(self._trace_callback)
    
    def get_write_queries(self) -> List[Tuple[str, Tuple]]:
        """
        Get only INSERT, UPDATE, DELETE queries (write operations).
        
        Note: set_trace_callback returns the fully expanded SQL with parameter
        values substituted in, so we return empty tuple for params.
        """
        write_queries = []
        for sql in self.queries:
            sql_upper = sql.strip().upper()
            if sql_upper.startswith(('INSERT', 'UPDATE', 'DELETE')):
                write_queries.append((sql, ()))
        return write_queries
    
    def get_read_queries(self) -> List[str]:
        """
        Get only SELECT queries (read operations).
        Used for read-only tasks that don't modify data.
        """
        read_queries = []
        for sql in self.queries:
            sql_upper = sql.strip().upper()
            if sql_upper.startswith('SELECT'):
                read_queries.append(sql)
        return read_queries
    
    def get_tables_from_read_queries(self) -> Set[str]:
        """
        Extract table names from SELECT queries.
        """
        tables = set()
        for sql in self.get_read_queries():
            # Pattern: SELECT ... FROM table_name
            matches = re.findall(r'FROM\s+(\w+)', sql, re.IGNORECASE)
            tables.update(matches)
            # Also check for JOINs
            join_matches = re.findall(r'JOIN\s+(\w+)', sql, re.IGNORECASE)
            tables.update(join_matches)
        return tables
    
    def get_all_queries(self) -> List[str]:
        """Get all logged queries."""
        return self.queries.copy()


class SQLQueryParser:
    """
    Static SQL query parser that analyzes INSERT/UPDATE/DELETE statements
    to identify affected tables, columns, and values.
    """
    
    @staticmethod
    def parse_insert(sql: str, params: Tuple) -> Optional[Dict[str, Any]]:
        """
        Parse an INSERT statement to extract table, columns, and values.
        
        Returns:
            Dict with keys: table, columns, values
        """
        # Pattern: INSERT INTO table (col1, col2, ...) VALUES (?, ?, ...)
        # or INSERT INTO table (col1, col2, ...) VALUES (val1, val2, ...)
        pattern = r'INSERT\s+(?:OR\s+\w+\s+)?INTO\s+(\w+)\s*\(([^)]+)\)\s*VALUES\s*\(([^)]+)\)'
        match = re.search(pattern, sql, re.IGNORECASE)
        
        if not match:
            # Try simpler pattern without column list
            pattern2 = r'INSERT\s+(?:OR\s+\w+\s+)?INTO\s+(\w+)\s+VALUES\s*\(([^)]+)\)'
            match2 = re.search(pattern2, sql, re.IGNORECASE)
            if match2:
                return {
                    'operation': 'INSERT',
                    'table': match2.group(1),
                    'columns': None,
                    'values': [c.strip() for c in match2.group(2).split(',')]
                }
            return None
        
        table = match.group(1)
        columns = [c.strip() for c in match.group(2).split(',')]
        
        # Map parameters to values
        values = [c.strip() for c in match.group(3).split(',')]
        
        return {
            'operation': 'INSERT',
            'table': table,
            'columns': columns,
            'values': values
        }
    
    @staticmethod
    def parse_update(sql: str, params: Tuple) -> Optional[Dict[str, Any]]:
        """
        Parse an UPDATE statement to extract table, columns, values, and conditions.
        
        Returns:
            Dict with keys: table, set_columns, set_values, where_columns, where_values
        """
        # Pattern: UPDATE table SET col1=?, col2=? WHERE col3=? AND col4=?
        pattern = r'UPDATE\s+(\w+)\s+SET\s+(.+?)(?:\s+WHERE\s+(.+))?$'
        match = re.search(pattern, sql, re.IGNORECASE | re.DOTALL)
        
        if not match:
            return None
        
        table = match.group(1)
        set_clause = match.group(2)
        where_clause = match.group(3) if match.group(3) else None
        
        # Parse SET clause
        set_parts = []
        for part in set_clause.split(','):
            part = part.strip()
            if '=' in part:
                col = part.split('=')[0].strip()
                set_parts.append(col)
        
        # Parse WHERE clause to extract column names
        where_columns = []
        if where_clause:
            # Simple extraction of column names from WHERE
            where_parts = re.findall(r'(\w+)\s*=', where_clause)
            where_columns = where_parts
        
        params_list = list(params) if params else []
        num_set = len(set_parts)
        
        return {
            'operation': 'UPDATE',
            'table': table,
            'set_columns': set_parts,
            'set_values': params_list[:num_set] if params_list else [],
            'where_columns': where_columns,
            'where_values': params_list[num_set:] if params_list else []
        }
    
    @staticmethod
    def parse_delete(sql: str, params: Tuple) -> Optional[Dict[str, Any]]:
        """
        Parse a DELETE statement to extract table and conditions.
        
        Returns:
            Dict with keys: table, where_columns, where_values
        """
        # Pattern: DELETE FROM table WHERE col1=? AND col2=?
        pattern = r'DELETE\s+FROM\s+(\w+)(?:\s+WHERE\s+(.+))?$'
        match = re.search(pattern, sql, re.IGNORECASE | re.DOTALL)
        
        if not match:
            return None
        
        table = match.group(1)
        where_clause = match.group(2) if match.group(2) else None
        
        where_columns = []
        if where_clause:
            where_parts = re.findall(r'(\w+)\s*=', where_clause)
            where_columns = where_parts
        
        return {
            'operation': 'DELETE',
            'table': table,
            'where_columns': where_columns,
            'where_values': list(params) if params else []
        }
    
    @staticmethod
    def parse_query(sql: str, params: Tuple) -> Optional[Dict[str, Any]]:
        """Parse any write query and return structured information."""
        sql_upper = sql.strip().upper()
        
        if sql_upper.startswith('INSERT'):
            return SQLQueryParser.parse_insert(sql, params)
        elif sql_upper.startswith('UPDATE'):
            return SQLQueryParser.parse_update(sql, params)
        elif sql_upper.startswith('DELETE'):
            return SQLQueryParser.parse_delete(sql, params)
        
        return None


class StaticVerifierGenerator:
    """
    Generates GLOBAL verifiers by analyzing the NET EFFECT of all executed SQL queries.
    
    The verifiers reflect the FINAL state of the database after ALL queries have executed,
    not intermediate states. For example:
    - INSERT 10; INSERT 20; DELETE 10 -> Verifier: 10 doesn't exist, 20 exists
    
    No LLM is used - purely based on SQL semantics and state comparison.
    """
    
    def __init__(self, db_schema: Dict[str, List[str]]):
        self.db_schema = db_schema
    
    def generate_verifiers(self, write_queries: List[Tuple[str, Tuple]], 
                          initial_state: Dict[str, List[Dict]], 
                          final_state: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """
        Generate GLOBAL verifiers based on the NET EFFECT of all SQL operations.
        
        This analyzes the sequence of all write queries to determine the final
        expected state, accounting for operations that may cancel each other out
        (e.g., INSERT followed by DELETE of the same row).
        
        Args:
            write_queries: List of (sql, params) tuples for write operations
            initial_state: Snapshot of database before execution
            final_state: Snapshot of database after execution
            
        Returns:
            List of verifier configurations for the FINAL state
        """
        # Parse all operations
        operations = []
        for sql, params in write_queries:
            parsed = SQLQueryParser.parse_query(sql, params)
            if parsed:
                operations.append(parsed)
        
        # Track net changes per table
        # Key: (table, row_identifier_tuple) -> 'exists' | 'deleted' | 'modified'
        net_changes = self._compute_net_changes(operations, initial_state, final_state)
        
        # Generate verifiers based on the FINAL state (not intermediate)
        verifiers = self._generate_verifiers_from_net_changes(net_changes, final_state)
        
        # If no verifiers from net changes, fall back to state diff
        if not verifiers:
            verifiers = self._generate_verifiers_from_state_diff(initial_state, final_state)
        
        # Deduplicate verifiers
        seen = set()
        unique_verifiers = []
        for v in verifiers:
            key = v.get('query', '')
            if key and key not in seen:
                seen.add(key)
                unique_verifiers.append(v)
        
        return unique_verifiers
    
    def _compute_net_changes(self, operations: List[Dict[str, Any]], 
                             initial_state: Dict[str, List[Dict]],
                             final_state: Dict[str, List[Dict]]) -> Dict[str, Dict[str, Any]]:
        """
        Compute the net effect of all operations on each row.
        
        This tracks INSERTs, UPDATEs, and DELETEs and determines the final
        state of each affected row, accounting for operations that cancel out.
        
        Returns:
            Dict mapping table names to their net changes
        """
        # Track changes per table: {table: {row_key: {'state': 'exists'|'deleted', 'data': {...}}}}
        net_changes = {}
        
        for op in operations:
            table = op.get('table', '')
            if not table:
                continue
            
            if table not in net_changes:
                net_changes[table] = {'inserts': [], 'updates': [], 'deletes': []}
            
            if op['operation'] == 'INSERT':
                columns = op.get('columns', [])
                values = op.get('values', [])
                if columns and values and len(columns) == len(values):
                    row_data = dict(zip(columns, values))
                    # Check if this row already marked for delete - if so, it now exists
                    row_key = self._get_row_key(table, row_data)
                    # Remove from deletes if present (INSERT after DELETE = exists)
                    net_changes[table]['deletes'] = [
                        d for d in net_changes[table]['deletes'] 
                        if self._get_row_key(table, d) != row_key
                    ]
                    net_changes[table]['inserts'].append(row_data)
            
            elif op['operation'] == 'UPDATE':
                set_columns = op.get('set_columns', [])
                set_values = op.get('set_values', [])
                where_columns = op.get('where_columns', [])
                where_values = op.get('where_values', [])
                
                if where_columns and where_values:
                    where_data = dict(zip(where_columns, where_values))
                    update_data = dict(zip(set_columns, set_values)) if set_columns and set_values else {}
                    net_changes[table]['updates'].append({
                        'where': where_data,
                        'set': update_data
                    })
            
            elif op['operation'] == 'DELETE':
                where_columns = op.get('where_columns', [])
                where_values = op.get('where_values', [])
                
                if where_columns and where_values:
                    delete_data = dict(zip(where_columns, where_values))
                    row_key = self._get_row_key(table, delete_data)
                    # Remove from inserts if present (INSERT then DELETE = doesn't exist)
                    net_changes[table]['inserts'] = [
                        i for i in net_changes[table]['inserts']
                        if self._get_row_key(table, i) != row_key
                    ]
                    # Only add to deletes if not in inserts (delete of pre-existing row)
                    net_changes[table]['deletes'].append(delete_data)
        
        return net_changes
    
    def _get_row_key(self, table: str, row_data: Dict[str, Any]) -> str:
        """Generate a unique key for a row based on its identifying columns."""
        # Use primary key columns if known, otherwise use all provided columns
        key_cols = []
        if table in self.db_schema:
            # Common primary key patterns
            for pk_candidate in ['id', f'{table}_id', 'chat_id', 'team_id', 'channel_id', 'user_id', 'message_id']:
                if pk_candidate in row_data:
                    key_cols.append(pk_candidate)
            
            # If composite key tables
            if table == 'chat_members':
                key_cols = ['chat_id', 'user_id']
            elif table == 'channel_members':
                key_cols = ['team_id', 'channel_id', 'user_id']
        
        if not key_cols:
            key_cols = list(row_data.keys())[:3]
        
        key_parts = [f"{col}={row_data.get(col, '')}" for col in key_cols if col in row_data]
        return f"{table}:{','.join(sorted(key_parts))}"
    
    def _generate_verifiers_from_net_changes(self, net_changes: Dict[str, Dict[str, Any]],
                                              final_state: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """
        Generate verifiers based on the computed net changes.
        
        These verifiers check the FINAL state after all operations complete.
        """
        verifiers = []
        
        for table, changes in net_changes.items():
            # Verifiers for rows that should EXIST (net inserts)
            for row_data in changes.get('inserts', []):
                v = self._build_existence_verifier(table, row_data, should_exist=True, final_state=final_state)
                if v:
                    verifiers.append(v)
            
            # Verifiers for rows that should be DELETED/NOT EXIST
            for delete_data in changes.get('deletes', []):
                v = self._build_existence_verifier(table, delete_data, should_exist=False, final_state=final_state)
                if v:
                    verifiers.append(v)
            
            # Verifiers for updated rows (check new values exist)
            for update in changes.get('updates', []):
                v = self._build_update_verifier(table, update, final_state)
                if v:
                    verifiers.append(v)
        
        return verifiers
    
    def _build_existence_verifier(self, table: str, row_data: Dict[str, Any], 
                                   should_exist: bool, final_state: Dict[str, List[Dict]]) -> Optional[Dict[str, Any]]:
        """Build a verifier to check if a row exists or doesn't exist."""
        where_parts = []
        for col, val in list(row_data.items())[:3]:
            if val is not None and col not in ('created_at', 'updated_at'):
                if isinstance(val, str):
                    # Escape single quotes in values
                    val = val.strip("'").strip('"')
                    escaped_val = val.replace("'", "''")
                    where_parts.append(f"{col}='{escaped_val}'")
                elif isinstance(val, (int, float)):
                    where_parts.append(f"{col}={val}")
        
        if not where_parts:
            return None
        
        where_clause = " AND ".join(where_parts)
        query = f"SELECT COUNT(*) FROM {table} WHERE {where_clause}"
        
        # Check if table has soft delete
        has_soft_delete = table in self.db_schema and 'is_soft_deleted' in self.db_schema[table]
        
        if should_exist:
            return {
                "query": query,
                "expected_value": 1,
                "comparison_type": "greater_than_or_equals",
                "failure_remark": f"Expected row in {table} was not found in final state."
            }
        else:
            if has_soft_delete:
                # For soft delete tables, check is_soft_deleted = 1
                query = f"SELECT COUNT(*) FROM {table} WHERE {where_clause} AND is_soft_deleted=1"
                return {
                    "query": query,
                    "expected_value": 1,
                    "comparison_type": "greater_than_or_equals",
                    "failure_remark": f"Expected row in {table} to be soft-deleted."
                }
            else:
                return {
                    "query": query,
                    "expected_value": 0,
                    "comparison_type": "equals",
                    "failure_remark": f"Expected row in {table} to be deleted but it still exists."
                }
    
    def _build_update_verifier(self, table: str, update: Dict[str, Any],
                                final_state: Dict[str, List[Dict]]) -> Optional[Dict[str, Any]]:
        """Build a verifier for an update operation (check row has new values)."""
        where_data = update.get('where', {})
        set_data = update.get('set', {})
        
        if not where_data:
            return None
        
        # Build WHERE clause combining identifying columns AND new values
        where_parts = []
        
        # Add identifying conditions
        for col, val in list(where_data.items())[:2]:
            if val is not None:
                if isinstance(val, str):
                    escaped_val = val.replace("'", "''")
                    where_parts.append(f"{col}='{escaped_val}'")
                elif isinstance(val, (int, float)):
                    where_parts.append(f"{col}={val}")
        
        # Add new values to verify
        for col, val in list(set_data.items())[:2]:
            if val is not None and col not in ('updated_at', 'created_at'):
                # Skip function calls like CURRENT_TIMESTAMP
                if isinstance(val, str) and '(' not in val:
                    escaped_val = val.replace("'", "''")
                    where_parts.append(f"{col}='{escaped_val}'")
                elif isinstance(val, (int, float)):
                    where_parts.append(f"{col}={val}")
        
        if not where_parts:
            return None
        
        where_clause = " AND ".join(where_parts)
        query = f"SELECT COUNT(*) FROM {table} WHERE {where_clause}"
        
        return {
            "query": query,
            "expected_value": 1,
            "comparison_type": "greater_than_or_equals",
            "failure_remark": f"Expected updated row in {table} with new values was not found."
        }
    
    def _generate_verifiers_from_state_diff(self, initial_state: Dict[str, List[Dict]], 
                                            final_state: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """
        Generate GLOBAL verifiers by comparing initial and final database states.
        
        This is the primary method - it directly compares what exists in the 
        final state vs initial state, giving us the TRUE net effect.
        """
        verifiers = []
        
        all_tables = set(list(initial_state.keys()) + list(final_state.keys()))
        
        for table in all_tables:
            initial_rows = initial_state.get(table, [])
            final_rows = final_state.get(table, [])
            
            # Create comparable row representations (exclude timestamp columns)
            def row_to_comparable(row: Dict) -> Tuple:
                items = [(k, v) for k, v in row.items() 
                        if k not in ('created_at', 'updated_at', 'modified_at')]
                return tuple(sorted(items))
            
            initial_set = set(row_to_comparable(r) for r in initial_rows if isinstance(r, dict))
            final_set = set(row_to_comparable(r) for r in final_rows if isinstance(r, dict))
            
            # NEW rows (in final but not in initial)
            new_rows = final_set - initial_set
            for row_tuple in list(new_rows)[:3]:  # Limit to 3 verifiers per table
                row = dict(row_tuple)
                v = self._build_existence_verifier(table, row, should_exist=True, final_state=final_state)
                if v:
                    verifiers.append(v)
            
            # DELETED rows (in initial but not in final)
            deleted_rows = initial_set - final_set
            for row_tuple in list(deleted_rows)[:2]:  # Limit to 2 delete verifiers
                row = dict(row_tuple)
                v = self._build_existence_verifier(table, row, should_exist=False, final_state=final_state)
                if v:
                    verifiers.append(v)
        
        return verifiers


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Verifier:
    """Represents a database state verifier."""
    verifier_type: str
    validation_config: Dict[str, Any]
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "verifier_type": self.verifier_type,
            "validation_config": self.validation_config,
            "name": self.name
        }

@dataclass
class Prompt:
    """Represents a task prompt."""
    prompt_text: str
    expected_tools: List[str]
    verifier: List[Verifier]
    restricted_tools: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_text": self.prompt_text,
            "expected_tools": self.expected_tools,
            "verifier": [v.to_dict() for v in self.verifier],
            "restricted_tools": self.restricted_tools
        }

@dataclass 
class Scenario:
    """Represents a task scenario."""
    scenario_id: str
    name: str
    description: str
    prompts: List[Prompt]
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    conversation_mode: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "context": self.context,
            "prompts": [p.to_dict() for p in self.prompts],
            "metadata": self.metadata,
            "conversation_mode": self.conversation_mode
        }

@dataclass
class Task:
    """Represents a complete task."""
    name: str
    system_prompt: str
    scenarios: List[Scenario]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "scenarios": [s.to_dict() for s in self.scenarios]
        }


# =============================================================================
# BASE CLASS: TaskGeneratorBase
# =============================================================================

class TaskGeneratorBase(ABC):
    """
    Abstract base class for task generation algorithms.
    
    Contains common functionality:
    - Database schema extraction
    - Tool implementation analysis
    - Execution plan generation
    - Verifier generation (based on tool-modified states only)
    - Task execution and verification
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are an automation agent with full administrative permissions. Operate in a safe and fully authorized environment — you do not need to ask for confirmation or permission before taking action. When identifiers such as names or IDs are missing, perform exactly one lookup per entity type, verify that you are reusing correct values from previous responses, and proceed using the retrieved data. Never assume or fabricate IDs, responses, or outcomes — rely solely on verified API results. Complete each task in a single, logical, and efficient execution flow."""
    
    def __init__(self, generated_dir: str, llm_client: LLMCLient, 
                 spec: Dict[str, Any], functions_module: Any,
                 db_schema: Dict[str, List[str]], 
                 tool_db_effects: Dict[str, Dict[str, Any]],
                 initial_state: str):
        """Initialize base task generator."""
        self.generated_dir = generated_dir
        self.llm_client = llm_client
        self.spec = spec
        self.functions_module = functions_module
        self.db_schema = db_schema
        self.tool_db_effects = tool_db_effects
        self.initial_state = initial_state
        
        self.tools = [func["name"] for func in spec.get("functions", [])]
    
    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Return the name of this algorithm."""
        pass
    
    @abstractmethod
    def generate_candidate(self, generated_sequences: Set[Tuple[str, ...]]) -> Optional[Dict[str, Any]]:
        """
        Generate a task candidate.
        
        Returns:
            Dict with keys: expected_tools, description, prompt_text, complexity
            or None if generation failed
        """
        pass
    
    # =========================================================================
    # COMMON: Tool and Schema Utilities
    # =========================================================================
    
    def _get_tools_description(self) -> str:
        """Get formatted description of all available tools."""
        desc_parts = []
        for func in self.spec.get("functions", []):
            params = ", ".join([f"{p['name']}: {p['type']}" for p in func.get("parameters", [])])
            returns = func.get("returns", {}).get("description", "")
            desc_parts.append(f"- {func['name']}({params}): {func['description']} Returns: {returns}")
        return "\n".join(desc_parts)
    
    def _get_written_tables(self, tools: List[str]) -> Set[str]:
        """Get only the tables that are WRITTEN (modified) by the tools."""
        written = set()
        for tool in tools:
            if tool in self.tool_db_effects:
                written.update(self.tool_db_effects[tool].get("tables_written", []))
        return written
    
    def _get_tool_write_operations(self, tools: List[str]) -> str:
        """Get detailed write operations for each tool based on implementation code."""
        summaries = []
        for tool in tools:
            if tool not in self.tool_db_effects:
                continue
            
            effect = self.tool_db_effects[tool]
            written_tables = effect.get("tables_written", [])
            
            if not written_tables:
                continue
            
            summary = f"TOOL: {tool}\n"
            summary += f"  Description: {effect.get('description', '')}\n"
            summary += f"  Modifies tables: {', '.join(written_tables)}\n"
            
            # Extract SQL write statements from implementation
            impl = effect.get("implementation", "")
            if impl:
                # Find INSERT statements
                inserts = re.findall(r'INSERT\s+INTO\s+(\w+)\s*\(([^)]+)\)', impl, re.IGNORECASE)
                for table, cols in inserts:
                    summary += f"  INSERT INTO {table} ({cols})\n"
                
                # Find UPDATE statements
                updates = re.findall(r'UPDATE\s+(\w+)\s+SET\s+([^W]+?)(?:WHERE|$)', impl, re.IGNORECASE | re.DOTALL)
                for table, sets in updates:
                    sets_clean = sets.strip()[:100]
                    summary += f"  UPDATE {table} SET {sets_clean}...\n"
                
                # Find DELETE statements
                deletes = re.findall(r'DELETE\s+FROM\s+(\w+)', impl, re.IGNORECASE)
                for table in deletes:
                    summary += f"  DELETE FROM {table}\n"
            
            summaries.append(summary)
        
        return "\n".join(summaries) if summaries else "No write operations detected."
    
    # =========================================================================
    # COMMON: Execution Plan Generation
    # =========================================================================
    
    def _get_tool_params_detailed(self, tool_names: List[str]) -> str:
        """Get detailed parameter info for specific tools."""
        lines = []
        for func in self.spec.get("functions", []):
            if func["name"] in tool_names:
                lines.append(f"\n{func['name']}:")
                lines.append(f"  Description: {func['description']}")
                lines.append(f"  Parameters:")
                for p in func.get("parameters", []):
                    req = "required" if "Optional" not in p.get("type", "") else "optional"
                    lines.append(f"    - {p['name']} ({p['type']}, {req}): {p.get('description', '')}")
        return "\n".join(lines)
    
    def generate_execution_plan(self, task_data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Generate execution plan with specific parameter values."""
        tools = task_data.get('expected_tools', [])
        
        system_prompt = """Generate tool calls with EXACT parameter names and values.

CRITICAL: Use the EXACT parameter names as shown in TOOL PARAMETERS section.
For example, use "chat-id" not "chatId", use "user@odata.bind" not "user_odata_bind".

Output JSON array:
[{"tool": "tool_name", "params": {"exact-param-name": "value"}}]

Return ONLY valid JSON array."""

        user_prompt = f"""Generate execution plan:

TASK: {task_data.get('prompt_text', '')}

TOOLS TO USE (in order): {tools}

TOOL PARAMETERS (use EXACT names):
{self._get_tool_params_detailed(tools)}

DATABASE (use these IDs and values):
{self.initial_state}

IMPORTANT: Use EXACT parameter names like "chat-id", "@odata.type", "user@odata.bind" as shown above."""

        try:
            response = self.llm_client.get_response_json(user_prompt, system_prompt)
            if response and isinstance(response, list):
                return response
        except:
            pass
        
        try:
            text_response = self.llm_client.get_response_text(user_prompt, system_prompt)
            match = re.search(r'\[[\s\S]*?\]', text_response)
            if match:
                return json.loads(match.group())
        except:
            pass
        return None
    
    # =========================================================================
    # COMMON: Verifier Generation (ONLY for tool-modified states)
    # =========================================================================
    
    def generate_verifiers(self, task_data: Dict[str, Any], 
                           execution_plan: List[Dict[str, Any]],
                           db_path: str = None) -> List[Dict[str, Any]]:
        """
        Generate RELIABLE verifiers by querying actual database state after execution.
        
        Strategy: Instead of asking LLM to guess what the state should be, we:
        1. Query the database to get actual values that were inserted/modified
        2. Generate verifiers that match the actual state
        3. This ensures verifiers always pass for correct executions
        """
        tools = task_data.get("expected_tools", [])
        written_tables = self._get_written_tables(tools)
        
        if not written_tables:
            return []
        
        verifiers = []
        
        # If we have db_path, generate verifiers from actual database state
        if db_path and os.path.exists(db_path):
            verifiers = self._generate_verifiers_from_db_state(
                tools, execution_plan, written_tables, db_path
            )
        
        # If we got verifiers from DB, return them
        if verifiers:
            return verifiers
        
        # Fallback to LLM-based generation (less reliable)
        return self._generate_verifiers_from_llm(task_data, execution_plan, written_tables)
    
    def _generate_verifiers_from_db_state(self, tools: List[str], 
                                           execution_plan: List[Dict[str, Any]],
                                           written_tables: Set[str],
                                           db_path: str) -> List[Dict[str, Any]]:
        """
        Generate verifiers by querying actual database state.
        
        This is the most reliable approach - we look at what actually changed
        in the database and create verifiers that match that state.
        """
        verifiers = []
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            for step in execution_plan:
                tool = step.get("tool", "")
                params = step.get("params", {})
                
                if tool not in self.tool_db_effects:
                    continue
                
                effect = self.tool_db_effects[tool]
                tool_written_tables = effect.get("tables_written", [])
                
                for table in tool_written_tables:
                    if table not in self.db_schema:
                        continue
                    
                    columns = self.db_schema[table]
                    
                    # Build WHERE conditions from params that match column names
                    conditions = []
                    param_values = []
                    for col in columns:
                        # Check various param naming conventions
                        param_keys = [col, col.replace("_", "-"), col.replace("_", ".")]
                        for pk in param_keys:
                            if pk in params:
                                conditions.append(f"{col} = ?")
                                param_values.append(params[pk])
                                break
                    
                    # Also check for common param patterns
                    param_col_mapping = {
                        "chat_id": ["chat-id", "chatId"],
                        "team_id": ["team-id", "teamId"],
                        "channel_id": ["channel-id", "channelId"],
                        "user_id": ["user-id", "userId", "user_odata_bind", "user@odata.bind"],
                        "odata_bind": ["user@odata.bind", "user_odata_bind"],
                        "body": ["body", "content", "message"],
                        "display_name": ["displayName", "display-name"],
                        "roles": ["roles", "role"],
                    }
                    
                    for col, param_names in param_col_mapping.items():
                        if col in columns and col not in [c.split(" = ")[0] for c in conditions]:
                            for pn in param_names:
                                if pn in params:
                                    conditions.append(f"{col} = ?")
                                    param_values.append(params[pn])
                                    break
                    
                    if conditions:
                        # Query to verify the row exists
                        where_clause = " AND ".join(conditions)
                        query = f"SELECT COUNT(*) FROM {table} WHERE {where_clause}"
                        
                        try:
                            cursor.execute(query, param_values)
                            count = cursor.fetchone()[0]
                            
                            if count > 0:
                                # Build a clean query with literal values for the verifier
                                clean_conditions = []
                                for i, cond in enumerate(conditions):
                                    col = cond.split(" = ")[0]
                                    val = param_values[i]
                                    if isinstance(val, str):
                                        clean_conditions.append(f"{col}='{val}'")
                                    else:
                                        clean_conditions.append(f"{col}={val}")
                                
                                clean_query = f"SELECT COUNT(*) FROM {table} WHERE {' AND '.join(clean_conditions)}"
                                
                                verifiers.append({
                                    "query": clean_query,
                                    "expected_value": count,
                                    "comparison_type": "equals",
                                    "failure_remark": f"Expected {count} row(s) in {table} matching the operation parameters."
                                })
                        except sqlite3.Error:
                            pass
            
            conn.close()
        except Exception:
            pass
        
        return verifiers
    
    def _generate_verifiers_from_llm(self, task_data: Dict[str, Any],
                                      execution_plan: List[Dict[str, Any]],
                                      written_tables: Set[str]) -> List[Dict[str, Any]]:
        """Fallback: Generate verifiers using LLM (less reliable)."""
        tools = task_data.get("expected_tools", [])
        write_operations = self._get_tool_write_operations(tools)
        
        schema_lines = ["SCHEMA (only tables modified by tools):"]
        for table_name in written_tables:
            if table_name in self.db_schema:
                schema_lines.append(f"  {table_name}: {', '.join(self.db_schema[table_name])}")
        written_schema = "\n".join(schema_lines)
        
        plan_desc = []
        for i, step in enumerate(execution_plan):
            tool = step.get("tool", "")
            params = step.get("params", {})
            plan_desc.append(f"{i+1}. {tool}({json.dumps(params)})")
        
        system_prompt = """Generate MINIMAL SQL verifiers to check if tool operations succeeded.

CRITICAL RULES:
1. Generate verifiers ONLY for data that the tools CREATE or MODIFY
2. Use the EXACT parameter values from the execution plan
3. Focus on INSERT operations - verify new rows exist
4. Generate 1 verifier per tool write operation (minimum needed)
5. Use COUNT(*) for existence checks
6. Use specific WHERE clauses matching the operation parameters

Output JSON array:
[{"query": "SELECT COUNT(*) FROM table WHERE col='value'", "expected_value": 1, "comparison_type": "equals", "failure_remark": "description"}]

Return ONLY valid JSON array."""

        user_prompt = f"""Generate verifiers for these tool operations:

EXECUTION PLAN (with exact parameters):
{chr(10).join(plan_desc)}

TOOL WRITE OPERATIONS:
{write_operations}

{written_schema}

Generate minimal verifiers using EXACT values from the execution plan."""

        try:
            response = self.llm_client.get_response_json(user_prompt, system_prompt)
            if response and isinstance(response, list):
                return response
        except:
            pass
        
        try:
            text_response = self.llm_client.get_response_text(user_prompt, system_prompt)
            match = re.search(r'\[[\s\S]*?\]', text_response)
            if match:
                return json.loads(match.group())
        except:
            pass
        return []
    
    # =========================================================================
    # COMMON: Execution and Verification
    # =========================================================================
    
    def execute_plan(self, plan: List[Dict[str, Any]], db_path: str) -> Tuple[bool, Optional[str]]:
        """Execute a plan against the database."""
        try:
            for step in plan:
                tool_name = step.get("tool", "")
                params = step.get("params", {})
                
                sanitized_params = {"db_path": db_path}
                for k, v in params.items():
                    san_k = k.replace("-", "_").replace(".", "_").replace("@", "").replace("$", "").replace("userodata", "user_odata")  
                    sanitized_params[san_k] = v
                
                if hasattr(self.functions_module, tool_name):
                    func = getattr(self.functions_module, tool_name)
                    func(**sanitized_params)
                else:
                    return False, f"Unknown tool: {tool_name}"
            
            return True, None
        except Exception as e:
            return False, str(e)
    
    def verify_task(self, verifiers: List[Dict[str, Any]], db_path: str) -> Tuple[bool, List[str]]:
        """Verify task completion by checking database state."""
        failures = []
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            for v in verifiers:
                config = v.get("validation_config", v)
                query = config.get("query", "")
                expected = config.get("expected_value")
                comparison = config.get("comparison_type", "equals")
                failure_msg = config.get("failure_remark", "Verification failed")
                
                try:
                    cursor.execute(query)
                    result = cursor.fetchone()
                    actual = result[0] if result else None
                    
                    passed = False
                    if comparison == "equals":
                        passed = actual == expected
                    elif comparison == "greater_than":
                        passed = actual is not None and actual > expected
                    elif comparison == "greater_than_or_equals":
                        passed = actual is not None and actual >= expected
                    elif comparison == "less_than":
                        passed = actual is not None and actual < expected
                    elif comparison == "less_than_or_equals":
                        passed = actual is not None and actual <= expected
                    
                    if not passed:
                        failures.append(f"{failure_msg} (expected: {expected}, got: {actual})")
                except sqlite3.Error as e:
                    failures.append(f"Query error: {e}")
            
            conn.close()
        except Exception as e:
            failures.append(f"Database error: {e}")
        
        return len(failures) == 0, failures
    
    # =========================================================================
    # COMMON: Task Object Creation
    # =========================================================================
    
    def create_task_object(self, task_data: Dict[str, Any], task_num: int) -> Task:
        """Create a Task object from task data."""
        verifiers = []
        for v in task_data.get("verifiers", []):
            verifiers.append(Verifier(
                verifier_type="database_state",
                validation_config={
                    "query": v.get("query", ""),
                    "expected_value": v.get("expected_value", 1),
                    "comparison_type": v.get("comparison_type", "equals"),
                    "failure_remark": v.get("failure_remark", "Verification failed")
                }
            ))
        
        prompt = Prompt(
            prompt_text=task_data.get("prompt_text", ""),
            expected_tools=task_data.get("expected_tools", []),
            verifier=verifiers
        )
        
        scenario = Scenario(
            scenario_id=f"s{task_num}_1",
            name=f"S{task_num}",
            description=task_data.get("description", ""),
            prompts=[prompt],
            context={},
            metadata={
                "difficulty": task_data.get("complexity", "medium"),
                "algorithm": self.algorithm_name
            }
        )
        
        return Task(
            name=f"task_{task_num}",
            system_prompt=self.DEFAULT_SYSTEM_PROMPT,
            scenarios=[scenario]
        )
    
    def generate_task_script(self, task_data: Dict[str, Any], task_num: int, 
                             plan: List[Dict[str, Any]], output_dir: str) -> str:
        """Generate a Python script for executing the task."""
        script_lines = [
            '"""',
            f'Task {task_num}: {task_data.get("description", "")}',
            f'Algorithm: {self.algorithm_name}',
            f'Prompt: {task_data.get("prompt_text", "")}',
            f'Tools: {task_data.get("expected_tools", [])}',
            '"""',
            '',
            'import os, sys, sqlite3',
            'sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))',
            f'from {os.path.basename(self.generated_dir)} import functions',
            '',
            'DB_PATH = os.path.join(os.path.dirname(__file__), "..", "function_state.db")',
            '',
            'def run_task():',
            f'    print("Running Task {task_num}")',
            '    functions.Reset(db_path=DB_PATH)',
            '    results = []',
        ]
        
        for i, step in enumerate(plan):
            tool_name = step.get("tool", "")
            params = step.get("params", {})
            sanitized_params = {}
            for k, v in params.items():
                san_k = k.replace("-", "_").replace(".", "_").replace("@", "").replace("$", "").replace("userodata", "user_odata")
                sanitized_params[san_k] = v
            
            script_lines.extend([
                f'    try:',
                f'        r{i} = functions.{tool_name}(**{repr(sanitized_params)}, db_path=DB_PATH)',
                f'        results.append(("ok", r{i}))',
                f'    except Exception as e:',
                f'        results.append(("err", str(e)))',
            ])
        
        script_lines.extend([
            '    return results',
            '',
            'def verify():',
            '    conn = sqlite3.connect(DB_PATH)',
            '    cur = conn.cursor()',
            '    verifiers = [',
        ])
        
        for v in task_data.get("verifiers", []):
            script_lines.append(f"        ({repr(v.get('query', ''))}, {repr(v.get('expected_value', 1))}, {repr(v.get('comparison_type', 'equals'))}),")
        
        script_lines.extend([
            '    ]',
            '    ok = True',
            '    for q, exp, cmp in verifiers:',
            '        try:',
            '            cur.execute(q)',
            '            act = cur.fetchone()[0] if cur.fetchone() else None',
            '            cur.execute(q)',
            '            act = cur.fetchone()[0] if cur.fetchone() else None',
            '            passed = (act == exp) if cmp == "equals" else (act > exp if cmp == "greater_than" else act < exp)',
            '            if not passed: ok = False',
            '        except: ok = False',
            '    conn.close()',
            '    return ok',
            '',
            'if __name__ == "__main__":',
            '    run_task()',
            '    print("PASS" if verify() else "FAIL")',
        ])
        
        script_path = os.path.join(output_dir, f"task_{task_num}.py")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(script_lines))
        return script_path


# =============================================================================
# DERIVED CLASS: ToolsFirstAlgorithm
# =============================================================================

class ToolsFirstAlgorithm(TaskGeneratorBase):
    """
    Tools-First Algorithm: Generate tool sequence first, then task description.
    
    Steps:
    1. Generate unique tool sequences
    2. Generate task description for the sequence
    3. Validate consistency with chain-of-thought
    """
    
    @property
    def algorithm_name(self) -> str:
        return "tools-first"
    
    def _get_known_sequences_summary(self, generated_sequences: Set[Tuple[str, ...]]) -> str:
        """Get summary of already generated tool sequences."""
        if not generated_sequences:
            return "No tasks generated yet."
        
        summaries = []
        for seq in list(generated_sequences)[:20]:
            summaries.append(f"- {' -> '.join(seq)}")
        
        result = "\n".join(summaries)
        if len(generated_sequences) > 20:
            result += f"\n... and {len(generated_sequences) - 20} more"
        return result
    
    def _generate_tool_sequences(self, batch_size: int, 
                                  generated_sequences: Set[Tuple[str, ...]]) -> List[Dict[str, Any]]:
        """Generate unique tool sequences, using strategy guidance if available."""
        
        # Build strategy guidance
        strategy_guidance = ""
        if hasattr(self, 'strategy') and self.strategy:
            # Priority tools guidance
            if self.strategy.prioritize_unused_tools:
                priority_tools = self.strategy.prioritize_unused_tools[:10]
                strategy_guidance += f"\nPRIORITY: Include these UNUSED tools in sequences: {priority_tools}"
            
            # Avoid overused tools
            if self.strategy.avoid_overused_tools:
                strategy_guidance += f"\nAVOID: These tools are overused, de-prioritize: {self.strategy.avoid_overused_tools}"
            
            # Length guidance
            min_len, max_len = self.strategy.get_length_range()
            strategy_guidance += f"\nTARGET LENGTH: Generate sequences with {min_len}-{max_len} tools."
            
            # Complexity guidance
            weights = self.strategy.get_complexity_weights()
            strategy_guidance += f"\nCOMPLEXITY MIX: {int(weights['simple']*100)}% simple, {int(weights['medium']*100)}% medium, {int(weights['complex']*100)}% complex."
        else:
            strategy_guidance = "\nMix: 30% simple (2 tools), 50% medium (3-4 tools), 20% complex (5+ tools)."
        
        system_prompt = """Generate UNIQUE sequences of tool calls that represent realistic workflows.

Rules:
1. Each sequence must be UNIQUE (different combination/order)
2. Sequences should be 2-5 tools long
3. Tools should work together logically
4. IMPORTANT: Follow the STRATEGY GUIDANCE to improve coverage

Output JSON array:
[{"tools": ["tool1", "tool2"], "workflow_type": "brief description", "complexity": "simple|medium|complex"}]

Return ONLY JSON array."""

        user_prompt = f"""Generate {batch_size} unique tool sequences:

TOOLS:
{self._get_tools_description()}

AVOID THESE SEQUENCES:
{self._get_known_sequences_summary(generated_sequences)}

STRATEGY GUIDANCE:{strategy_guidance}"""

        response = self.llm_client.get_response_json(user_prompt, system_prompt)
        if response and isinstance(response, list):
            return response
        
        try:
            text_response = self.llm_client.get_response_text(user_prompt, system_prompt)
            match = re.search(r'\[[\s\S]*\]', text_response)
            if match:
                return json.loads(match.group())
        except:
            pass
        return []
    
    def _generate_description(self, tool_sequence: List[str], workflow_type: str) -> Optional[Dict[str, Any]]:
        """Generate SHORT task description for tool sequence."""
        system_prompt = """Write a CONCISE task description (1-3 sentences) for the given tools.

Rules:
1. Be BRIEF - 1 to 3 sentences only
2. Do NOT mention tool names
3. Mention specific parameter names/values (e.g, person name, person email address)
4. Reference actual data from the database
5. Describe WHAT to do, not HOW

Output JSON: {"description": "one-line summary", "prompt_text": "1-3 sentence task"}
Return ONLY JSON."""

        user_prompt = f"""Write a 1-3 sentence task for these tools: {tool_sequence}
Workflow: {workflow_type}

DATABASE DATA:
{self.initial_state}

Be concise!"""

        try:
            response = self.llm_client.get_response_json(user_prompt, system_prompt)
            if response and isinstance(response, dict):
                return response
        except:
            pass
        return None
    
    def _validate_consistency(self, task_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate with chain-of-thought reasoning."""
        system_prompt = """Validate if the task description REQUIRES exactly the specified tools.

Use chain-of-thought reasoning:
1. What actions does the task require?
2. Which tool handles each action?
3. Are all tools necessary?
4. Are the tools sufficient?

Output JSON:
{
    "reasoning": "step-by-step analysis",
    "is_consistent": true/false,
    "reason": "brief conclusion"
}
Return ONLY JSON."""

        user_prompt = f"""Validate this task-tool pair:

TASK: {task_data.get('prompt_text', '')}
TOOLS: {task_data.get('expected_tools', [])}

TOOL DESCRIPTIONS:
{self._get_tools_description()}

Think step-by-step."""

        try:
            response = self.llm_client.get_response_json(user_prompt, system_prompt)
            if response and isinstance(response, dict):
                return response.get("is_consistent", False), response.get("reason", "Unknown")
        except:
            pass
        return True, "Validation skipped"
    
    def generate_candidate(self, generated_sequences: Set[Tuple[str, ...]]) -> Optional[Dict[str, Any]]:
        """Generate a task candidate using tools-first approach."""
        sequences = self._generate_tool_sequences(5, generated_sequences)
        
        for seq in sequences:
            tools = seq.get("tools", [])
            if not tools or tuple(tools) in generated_sequences:
                continue
            
            desc_data = self._generate_description(tools, seq.get("workflow_type", ""))
            if not desc_data:
                continue
            
            task_data = {
                "expected_tools": tools,
                "complexity": seq.get("complexity", "medium"),
                "description": desc_data.get("description", ""),
                "prompt_text": desc_data.get("prompt_text", "")
            }
            
            is_consistent, reason = self._validate_consistency(task_data)
            if not is_consistent:
                print(f"    Inconsistent: {reason[:40]}...")
                continue
            
            return task_data
        
        return None


# =============================================================================
# DERIVED CLASS: TaskFirstAlgorithm
# =============================================================================

class TaskFirstAlgorithm(TaskGeneratorBase):
    """
    Task-First Algorithm: Generate task description first, then identify tools.
    
    Steps:
    1. Generate task descriptions. Use specific names/values that can be used to fill tool parameters. 
    2. Identify tools needed for each task
    3. Validate consistency with chain-of-thought
    """
    
    @property
    def algorithm_name(self) -> str:
        return "task-first"
    
    def _get_known_sequences_summary(self, generated_sequences: Set[Tuple[str, ...]]) -> str:
        """Get summary of already generated tool sequences to avoid duplicates."""
        if not generated_sequences:
            return "No tasks generated yet - feel free to propose any workflow."
        
        summaries = []
        for seq in list(generated_sequences)[:25]:
            summaries.append(f"- {' -> '.join(seq)}")
        
        result = "\n".join(summaries)
        if len(generated_sequences) > 25:
            result += f"\n... and {len(generated_sequences) - 25} more sequences"
        return result
    
    def _generate_task_descriptions(self, batch_size: int, 
                                     generated_sequences: Set[Tuple[str, ...]] = None) -> List[Dict[str, Any]]:
        """Generate SHORT task descriptions that lead to NOVEL tool sequences."""
        
        # Build guidance about known sequences
        sequences_guidance = ""
        if generated_sequences:
            sequences_guidance = f"""
IMPORTANT - AVOID THESE ALREADY-USED TOOL SEQUENCES:
{self._get_known_sequences_summary(generated_sequences)}

Generate tasks that would require DIFFERENT tool combinations than those listed above.
For example, if most tasks use "CreateTeam -> AddMember", create tasks that use other patterns like:
- Single tool operations (list, get, search)
- Different tool combinations (delete, update operations)
- Longer workflows with 3-5 tools
"""
        
        # Build strategy guidance if available
        strategy_guidance = ""
        if hasattr(self, 'strategy') and self.strategy:
            if self.strategy.prioritize_unused_tools:
                priority_tools = self.strategy.prioritize_unused_tools[:8]
                strategy_guidance += f"\nPRIORITY: Generate tasks that would use these UNUSED tools: {priority_tools}"
            
            if self.strategy.avoid_overused_tools:
                strategy_guidance += f"\nAVOID: These tools are overused: {self.strategy.avoid_overused_tools}"
            
            weights = self.strategy.get_complexity_weights()
            strategy_guidance += f"\nCOMPLEXITY TARGET: {int(weights['simple']*100)}% simple, {int(weights['medium']*100)}% medium, {int(weights['complex']*100)}% complex."
        
        system_prompt = """Generate CONCISE task descriptions (1-3 sentences each) for automation tasks.

Rules:
1. Each task should be 1-3 sentences
2. Be specific - reference actual data
3. Tasks should be achievable with the available tools
4. Do NOT mention tool names
5. Mention specific parameter names/values (e.g, person name, person email address) that can be used to fill tool parameters
6. CRITICAL: Generate tasks that would require NOVEL/UNIQUE tool sequences, not duplicates of existing ones

Output JSON array:
[{"description": "one-line summary", "prompt_text": "1-3 sentence task"}]

Return ONLY JSON array."""

        user_prompt = f"""Generate {batch_size} diverse, BRIEF task descriptions.

AVAILABLE TOOLS:
{self._get_tools_description()}

DATABASE DATA:
{self.initial_state}
{sequences_guidance}
{strategy_guidance}

Tasks should be realistic business scenarios that lead to UNIQUE tool combinations. Be concise!"""

        response = self.llm_client.get_response_json(user_prompt, system_prompt)
        if response and isinstance(response, list):
            return response
        
        try:
            text_response = self.llm_client.get_response_text(user_prompt, system_prompt)
            match = re.search(r'\[[\s\S]*\]', text_response)
            if match:
                return json.loads(match.group())
        except:
            pass
        return []
    
    def _identify_tools(self, task_description: str) -> Optional[List[str]]:
        """Identify tools needed for the task."""
        system_prompt = """Identify the MINIMUM set of tools needed to complete the task.

Use chain-of-thought:
1. What actions does the task require?
2. Which tool handles each action?
3. What order should tools be called?

Output JSON:
{
    "reasoning": "step-by-step analysis",
    "tools": ["tool1", "tool2"]
}
Return ONLY JSON."""

        user_prompt = f"""What tools are needed for this task?

TASK: {task_description}

AVAILABLE TOOLS:
{self._get_tools_description()}

Identify the minimum tools in correct order."""

        try:
            response = self.llm_client.get_response_json(user_prompt, system_prompt)
            if response and isinstance(response, dict):
                return response.get("tools", [])
        except:
            pass
        return None
    
    def _validate_consistency(self, task_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate with chain-of-thought reasoning."""
        system_prompt = """Validate if the identified tools can complete the task.

Use chain-of-thought:
1. Parse each requirement in the task
2. Map each requirement to a tool
3. Check if tools cover ALL requirements
4. Check if any tool is unnecessary

Output JSON:
{
    "reasoning": "step-by-step analysis",
    "is_consistent": true/false,
    "reason": "brief conclusion"
}
Return ONLY JSON."""

        user_prompt = f"""Validate tools for this task:

TASK: {task_data.get('prompt_text', '')}
IDENTIFIED TOOLS: {task_data.get('expected_tools', [])}

TOOL DESCRIPTIONS:
{self._get_tools_description()}

Think step-by-step."""

        try:
            response = self.llm_client.get_response_json(user_prompt, system_prompt)
            if response and isinstance(response, dict):
                return response.get("is_consistent", False), response.get("reason", "Unknown")
        except:
            pass
        return True, "Validation skipped"
    
    def generate_candidate(self, generated_sequences: Set[Tuple[str, ...]]) -> Optional[Dict[str, Any]]:
        """Generate a task candidate using task-first approach."""
        # Pass generated_sequences so LLM knows what to avoid
        tasks = self._generate_task_descriptions(5, generated_sequences)
        
        for task in tasks:
            prompt_text = task.get("prompt_text", "")
            if not prompt_text:
                continue
            
            tools = self._identify_tools(prompt_text)
            if not tools or tuple(tools) in generated_sequences:
                continue
            
            task_data = {
                "expected_tools": tools,
                "complexity": "medium",
                "description": task.get("description", ""),
                "prompt_text": prompt_text
            }
            
            is_consistent, reason = self._validate_consistency(task_data)
            if not is_consistent:
                print(f"    Inconsistent: {reason[:40]}...")
                continue
            
            return task_data
        
        return None


# =============================================================================
# MAIN CLASS: TaskGenerator
# =============================================================================

@dataclass
class GenerationStrategy:
    """
    Strategy for task generation based on stats analysis.
    
    Contains guidance on what types of tasks to prioritize.
    """
    prioritize_unused_tools: List[str] = field(default_factory=list)
    target_sequence_length: str = "medium"  # "short", "medium", "long"
    target_complexity: str = "medium"  # "simple", "medium", "complex"
    avoid_overused_tools: List[str] = field(default_factory=list)
    coverage_gap: float = 0.0  # How much coverage is missing (0-1)
    
    def get_length_range(self) -> Tuple[int, int]:
        """Get target sequence length range."""
        if self.target_sequence_length == "short":
            return (2, 2)
        elif self.target_sequence_length == "long":
            return (4, 6)
        else:  # medium
            return (3, 4)
    
    def get_complexity_weights(self) -> Dict[str, float]:
        """Get complexity distribution weights."""
        if self.target_complexity == "simple":
            return {"simple": 0.6, "medium": 0.3, "complex": 0.1}
        elif self.target_complexity == "complex":
            return {"simple": 0.1, "medium": 0.3, "complex": 0.6}
        else:  # medium
            return {"simple": 0.3, "medium": 0.5, "complex": 0.2}

class TaskGenerator:
    """
    Main task generator that orchestrates both algorithms.
    
    Uses:
    - ToolsFirstAlgorithm: Generate tool sequence first, then description
    - TaskFirstAlgorithm: Generate task description first, then identify tools
    
    Can load stats.json to improve task generation by:
    - Prioritizing unused tools
    - Adjusting sequence length distribution
    - Balancing complexity
    """
    
    def __init__(self, generated_dir: str):
        """Initialize the task generator."""
        self.generated_dir = os.path.abspath(generated_dir)
        self.llm_client = LLMCLient()
        
        # Load spec
        spec_path = os.path.join(generated_dir, "spec.json")
        with open(spec_path, 'r', encoding='utf-8') as f:
            self.spec = json.load(f)
        
        # Load functions module
        self._load_functions_module()
        
        # Extract database schema and tool effects
        self.db_schema = self._extract_db_schema()
        self.initial_state = self._get_initial_state_description()
        self.tool_db_effects = self._extract_tool_db_effects()
        
        # Track generated tool sequences for uniqueness
        self.generated_sequences: Set[Tuple[str, ...]] = set()
        
        # Load stats and compute generation strategy
        self.stats = self._load_stats()
        self.strategy = self._compute_generation_strategy()
        
        # Initialize both algorithms with strategy
        common_args = {
            "generated_dir": self.generated_dir,
            "llm_client": self.llm_client,
            "spec": self.spec,
            "functions_module": self.functions_module,
            "db_schema": self.db_schema,
            "tool_db_effects": self.tool_db_effects,
            "initial_state": self.initial_state
        }
        
        self.tools_first = ToolsFirstAlgorithm(**common_args)
        self.task_first = TaskFirstAlgorithm(**common_args)
        
        # Pass strategy to algorithms
        self.tools_first.strategy = self.strategy
        self.task_first.strategy = self.strategy
    
    def _load_stats(self) -> Optional[Dict[str, Any]]:
        """Load stats.json if it exists in the generated directory."""
        stats_path = os.path.join(self.generated_dir, "stats.json")
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                print(f"Loaded stats from {stats_path}")
                return stats
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load stats.json: {e}")
        return None
    
    def _compute_generation_strategy(self) -> GenerationStrategy:
        """
        Compute generation strategy based on loaded stats.
        
        Analyzes weaknesses in current task set and creates a strategy
        to address them.
        """
        strategy = GenerationStrategy()
        
        if not self.stats:
            print("No stats available - using default generation strategy")
            return strategy
        
        # 1. Identify unused tools to prioritize
        tool_coverage = self.stats.get("tool_coverage", {})
        unused_tools = tool_coverage.get("unused_tools", [])
        if unused_tools:
            strategy.prioritize_unused_tools = unused_tools
            print(f"Strategy: Prioritizing {len(unused_tools)} unused tools")
        
        # Calculate coverage gap
        coverage_pct = tool_coverage.get("coverage_percentage", 100)
        strategy.coverage_gap = (100 - coverage_pct) / 100
        
        # 2. Identify overused tools to avoid
        tool_frequency = self.stats.get("tool_frequency", {})
        if tool_frequency:
            avg_usage = sum(tool_frequency.values()) / len(tool_frequency) if tool_frequency else 0
            overused = [tool for tool, count in tool_frequency.items() if count > avg_usage * 2]
            if overused:
                strategy.avoid_overused_tools = overused[:5]  # Top 5 overused
                print(f"Strategy: De-prioritizing {len(strategy.avoid_overused_tools)} overused tools")
        
        # 3. Adjust sequence length based on current distribution
        seq_length = self.stats.get("sequence_length", {})
        avg_length = seq_length.get("average", 3)
        
        if avg_length < 2.5:
            strategy.target_sequence_length = "long"
            print("Strategy: Targeting longer sequences (current avg too short)")
        elif avg_length > 4.5:
            strategy.target_sequence_length = "short"
            print("Strategy: Targeting shorter sequences (current avg too long)")
        else:
            strategy.target_sequence_length = "medium"
        
        # 4. Adjust complexity based on current distribution
        complexity_dist = self.stats.get("complexity_distribution", {})
        if complexity_dist:
            total = sum(complexity_dist.values())
            simple_pct = complexity_dist.get("simple", 0) / total if total > 0 else 0
            complex_pct = complexity_dist.get("complex", 0) / total if total > 0 else 0
            
            if simple_pct > 0.5:
                strategy.target_complexity = "complex"
                print("Strategy: Targeting complex tasks (too many simple)")
            elif complex_pct > 0.4:
                strategy.target_complexity = "simple"
                print("Strategy: Targeting simple tasks (too many complex)")
            else:
                strategy.target_complexity = "medium"
        
        return strategy
    
    def _load_functions_module(self) -> None:
        """Load the functions module dynamically."""
        functions_path = os.path.join(self.generated_dir, "functions.py")
        spec = importlib.util.spec_from_file_location("functions", functions_path)
        self.functions_module = importlib.util.module_from_spec(spec)
        sys.modules["functions"] = self.functions_module
        spec.loader.exec_module(self.functions_module)
    
    def _extract_db_schema(self) -> Dict[str, List[str]]:
        """Extract database schema by running Reset and inspecting tables."""
        temp_db = os.path.join(self.generated_dir, "_temp_schema.db")
        schema = {}
        conn = None
        
        try:
            self.functions_module.Reset(db_path=temp_db)
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            for (table_name,) in tables:
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [col[1] for col in cursor.fetchall()]
                schema[table_name] = columns
            
            return schema
        finally:
            if conn:
                conn.close()
            import time
            time.sleep(0.1)
            try:
                if os.path.exists(temp_db):
                    os.remove(temp_db)
            except PermissionError:
                pass
    
    def _get_initial_state_description(self) -> str:
        """Extract description of initial database state."""
        temp_db = os.path.join(self.generated_dir, "_temp_task_gen.db")
        conn = None
        try:
            self.functions_module.Reset(db_path=temp_db)
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            state_desc = []
            for (table_name,) in tables:
                cursor.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()
                if rows:
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [col[1] for col in cursor.fetchall()]
                    
                    state_desc.append(f"\n{table_name} (columns: {', '.join(columns)}):")
                    for row in rows[:10]:
                        row_dict = dict(zip(columns, row))
                        state_desc.append(f"  - {row_dict}")
            
            return "\n".join(state_desc)
        finally:
            if conn:
                conn.close()
            import time
            time.sleep(0.1)
            try:
                if os.path.exists(temp_db):
                    os.remove(temp_db)
            except PermissionError:
                pass
    
    def _extract_tool_db_effects(self) -> Dict[str, Dict[str, Any]]:
        """Extract database effects for each tool by analyzing function implementations."""
        effects = {}
        functions_path = os.path.join(self.generated_dir, "functions.py")
        
        try:
            with open(functions_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except:
            return effects
        
        for func in self.spec.get("functions", []):
            tool_name = func["name"]
            effects[tool_name] = {
                "tables_read": set(),
                "tables_written": set(),
                "description": func.get("description", ""),
                "implementation": ""
            }
            
            pattern = rf'def {tool_name}\([^)]*\)[^:]*:(.*?)(?=\ndef |\Z)'
            match = re.search(pattern, source_code, re.DOTALL)
            
            if match:
                func_body = match.group(1)
                effects[tool_name]["implementation"] = func_body[:1000].strip()
                
                for table in self.db_schema.keys():
                    if re.search(rf'SELECT\s+.*FROM\s+{table}', func_body, re.IGNORECASE):
                        effects[tool_name]["tables_read"].add(table)
                    if re.search(rf'FROM\s+{table}\b', func_body, re.IGNORECASE):
                        effects[tool_name]["tables_read"].add(table)
                    if re.search(rf'INSERT\s+INTO\s+{table}', func_body, re.IGNORECASE):
                        effects[tool_name]["tables_written"].add(table)
                    if re.search(rf'UPDATE\s+{table}\b', func_body, re.IGNORECASE):
                        effects[tool_name]["tables_written"].add(table)
                    if re.search(rf'DELETE\s+FROM\s+{table}', func_body, re.IGNORECASE):
                        effects[tool_name]["tables_written"].add(table)
            
            effects[tool_name]["tables_read"] = list(effects[tool_name]["tables_read"])
            effects[tool_name]["tables_written"] = list(effects[tool_name]["tables_written"])
        
        return effects
    
    def _take_db_snapshot(self, db_path: str) -> Dict[str, List[Dict]]:
        """Take a snapshot of the current database state."""
        snapshot = {}
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            for (table_name,) in tables:
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [col[1] for col in cursor.fetchall()]
                
                cursor.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()
                
                snapshot[table_name] = [dict(zip(columns, row)) for row in rows]
            
            conn.close()
        except Exception as e:
            print(f"      Warning: Failed to take snapshot: {e}")
        
        return snapshot
    
    def _generate_read_only_verifiers(self, read_tables: Set[str], db_path: str) -> List[Dict[str, Any]]:
        """
        Generate verifiers for read-only tasks (tasks that don't modify data).
        
        For read-only operations, we verify that the tables being read contain
        the expected number of rows (based on current state).
        """
        verifiers = []
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            for table in read_tables:
                if table not in self.db_schema:
                    continue
                
                # Count rows in the table
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                
                if count > 0:
                    verifiers.append({
                        "query": f"SELECT COUNT(*) FROM {table}",
                        "expected_value": count,
                        "comparison_type": "greater_than_or_equals",
                        "failure_remark": f"Expected at least {count} rows in {table} for read operation."
                    })
            
            conn.close()
        except Exception as e:
            print(f"      Warning: Failed to generate read-only verifiers: {e}")
        
        return verifiers
    
    def _execute_plan_with_tracing(self, algorithm: TaskGeneratorBase, 
                                    plan: List[Dict[str, Any]], 
                                    db_path: str,
                                    sql_logger: SQLQueryLogger) -> Tuple[bool, Optional[str]]:
        """
        Execute a plan with SQL tracing enabled via set_trace_callback.
        
        Uses a patched sqlite3.connect to automatically enable tracing on all connections.
        """
        original_connect = sqlite3.connect
        
        def traced_connect(*args, **kwargs):
            """Wrapper that sets up trace callback on new connections."""
            conn = original_connect(*args, **kwargs)
            conn.set_trace_callback(sql_logger._trace_callback)
            return conn
        
        # Patch sqlite3.connect temporarily
        sqlite3.connect = traced_connect
        sql_logger._is_logging = True
        
        try:
            success, error = algorithm.execute_plan(plan, db_path)
            return success, error
        finally:
            # Restore original connect
            sqlite3.connect = original_connect
            sql_logger._is_logging = False
    
    def _process_task_candidate(self, algorithm: TaskGeneratorBase, 
                                 task_data: Dict[str, Any], 
                                 task_num: int, output_dir: str) -> Optional[str]:
        """
        Process a task candidate through execution and verification.
        
        Flow:
        1. Take initial snapshot of database state
        2. Enable SQL query logging using set_trace_callback
        3. Execute the generated plan
        4. Take final state of the database
        5. Statically parse executed SQL queries to generate verifiers
        """
        # Generate execution plan
        plan = algorithm.generate_execution_plan(task_data)
        if not plan:
            print("      Failed to generate plan")
            return None
        
        # Execute and test with SQL logging
        temp_db = os.path.join(self.generated_dir, f"_temp_task_{task_num}.db")
        sql_logger = SQLQueryLogger(temp_db)
        
        try:
            # Step 1: Reset database and take initial snapshot
            self.functions_module.Reset(db_path=temp_db)
            initial_state = self._take_db_snapshot(temp_db)
            
            # Step 2 & 3: Execute plan with SQL tracing enabled via set_trace_callback
            success, error = self._execute_plan_with_tracing(algorithm, plan, temp_db, sql_logger)
            
            if not success:
                print(f"      Execution failed: {error if error else 'Unknown'}...")
                return None
            
            # Step 4: Take final snapshot of database state
            final_state = self._take_db_snapshot(temp_db)
            
            # Step 5: Generate verifiers from SQL query analysis (NO LLM)
            write_queries = sql_logger.get_write_queries()
            
            if write_queries:
                # Use static verifier generator for write operations
                static_gen = StaticVerifierGenerator(self.db_schema)
                verifiers = static_gen.generate_verifiers(write_queries, initial_state, final_state)
            else:
                # Try to generate verifiers from state diff first
                static_gen = StaticVerifierGenerator(self.db_schema)
                verifiers = static_gen._generate_verifiers_from_state_diff(initial_state, final_state)
            
            # For read-only tasks (no write queries and no state diff), generate count verifiers
            # based on the tables accessed by SELECT queries
            if not verifiers:
                read_tables = sql_logger.get_tables_from_read_queries()
                if read_tables:
                    verifiers = self._generate_read_only_verifiers(read_tables, temp_db)
            
            if not verifiers:
                print("      Failed to generate verifiers (no operations detected)")
                return None
            
            task_data["verifiers"] = verifiers
            
            # Verify the task (should pass since verifiers are based on actual state)
            passed, failures = algorithm.verify_task(verifiers, temp_db)
            if not passed:
                print(f"      Verification failed: {failures[0] if failures else 'Unknown'}...")
                return None
            
            # Success! Save task
            self.generated_sequences.add(tuple(task_data["expected_tools"]))
            task = algorithm.create_task_object(task_data, task_num)
            
            task_path = os.path.join(output_dir, f"task_{task_num}.json")
            with open(task_path, 'w', encoding='utf-8') as f:
                json.dump(task.to_dict(), f, indent=2)
            
            algorithm.generate_task_script(task_data, task_num, plan, output_dir)
            return task_path
            
        except Exception as e:
            print(f"      Error during processing: {str(e)}...")
            return None
            
        finally:
            import time, gc
            gc.collect()
            time.sleep(0.1)
            try:
                if os.path.exists(temp_db):
                    os.remove(temp_db)
            except PermissionError:
                pass
    
    def _load_existing_tasks(self, output_dir: str) -> Tuple[Set[Tuple[str, ...]], int]:
        """
        Load existing tasks from the output directory to enable incremental generation.
        
        Returns:
            Tuple of (set of existing tool sequences, highest task number)
        """
        existing_sequences: Set[Tuple[str, ...]] = set()
        max_task_num = 0
        
        if not os.path.exists(output_dir):
            return existing_sequences, max_task_num
        
        # Find all existing task JSON files
        for filename in os.listdir(output_dir):
            if filename.startswith("task_") and filename.endswith(".json"):
                try:
                    # Extract task number
                    task_num_str = filename[5:-5]  # Remove "task_" and ".json"
                    task_num = int(task_num_str)
                    max_task_num = max(max_task_num, task_num)
                    
                    # Load the task file and extract tool sequences
                    task_path = os.path.join(output_dir, filename)
                    with open(task_path, 'r', encoding='utf-8') as f:
                        task_data = json.load(f)
                    
                    # Extract expected_tools from the task structure
                    for scenario in task_data.get("scenarios", []):
                        for prompt in scenario.get("prompts", []):
                            expected_tools = prompt.get("expected_tools", [])
                            if expected_tools:
                                existing_sequences.add(tuple(expected_tools))
                
                except (ValueError, json.JSONDecodeError, KeyError) as e:
                    print(f"  Warning: Could not load {filename}: {e}")
                    continue
        
        return existing_sequences, max_task_num
    
    def generate_tasks(self, num_tasks: int = 50, output_dir: str = None) -> List[str]:
        """
        Generate tasks using BOTH algorithms alternately.
        
        Supports incremental generation: loads existing tasks from output_dir
        and avoids generating tasks with duplicate tool sequences.
        """
        if output_dir is None:
            output_dir = os.path.join(self.generated_dir, "tasks")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load existing tasks for incremental generation
        existing_sequences, last_task_num = self._load_existing_tasks(output_dir)
        self.generated_sequences.update(existing_sequences)
        
        generated_files = []
        task_num = last_task_num + 1  # Start from the next task number
        attempts = 0
        max_attempts = num_tasks * 10
        use_tools_first = True
        
        tools = [func["name"] for func in self.spec.get("functions", [])]
        print(f"Generating {num_tasks} tasks using both algorithms...")
        print(f"Tools: {len(tools)}, Tables: {list(self.db_schema.keys())}")
        
        if existing_sequences:
            print(f"Loaded {len(existing_sequences)} existing tool sequences (will skip duplicates)")
            print(f"Starting from task_{task_num}")
        
        while len(generated_files) < num_tasks and attempts < max_attempts:
            algorithm = self.tools_first if use_tools_first else self.task_first
            alg_name = algorithm.algorithm_name.upper()
            
            print(f"\n[{len(generated_files)+1}/{num_tasks}] Using: {alg_name}")
            
            task_data = algorithm.generate_candidate(self.generated_sequences)
            
            if task_data:
                print(f"  Tools: {task_data.get('expected_tools', [])}")
                result = self._process_task_candidate(algorithm, task_data, task_num, output_dir)
                if result:
                    generated_files.append(result)
                    print(f"  ✓ Saved task_{task_num}")
                    task_num += 1
            else:
                print("  No valid candidate generated")
            
            attempts += 1
            use_tools_first = not use_tools_first
        
        print(f"\n{'='*60}")
        print(f"Generated {len(generated_files)} tasks")
        print(f"Output: {output_dir}")
        
        return generated_files


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate tasks using dual algorithms")
    parser.add_argument("--generated-dir", default="generated_teams", 
                       help="Directory containing generated functions")
    parser.add_argument("--output-dir", default=None,
                       help="Directory to save generated tasks")
    parser.add_argument("--num-tasks", type=int, default=50,
                       help="Number of tasks to generate")
    
    args = parser.parse_args()
    
    generator = TaskGenerator(args.generated_dir)
    generator.generate_tasks(num_tasks=args.num_tasks, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
