"""
SQLite Utility Functions

This module provides a centralized way to add custom functions to SQLite connections.
All utility functions are registered through a single function call.

Usage:
    import sqlite3
    from sqlite_utils import add_sqlite_functions
    
    conn = sqlite3.connect('your_database.db')
    add_sqlite_functions(conn)
    

    cursor = conn.execute("SELECT regexp_replace(column, '[^a-zA-Z0-9 ]', '', 'g') FROM table")
"""

import re
import sqlite3
import logging
import math
import json
from datetime import datetime, timedelta
from typing import Optional, Union, Any

logger = logging.getLogger(__name__)





def regexp_replace(string: str, pattern: str, replacement: str, flags: Optional[str] = None) -> str:
    """SQLite regexp_replace function implementation"""
    try:
        if string is None:
            return ""
        
        string = str(string)
        pattern = str(pattern) if pattern is not None else ""
        replacement = str(replacement) if replacement is not None else ""
        
        if not pattern:
            return string
        
        re_flags = 0
        if flags:
            flags = str(flags).lower()
            if 'i' in flags:
                re_flags |= re.IGNORECASE
            if 'm' in flags:
                re_flags |= re.MULTILINE
            if 's' in flags:
                re_flags |= re.DOTALL
        
        return re.sub(pattern, replacement, string, flags=re_flags)
        
    except Exception as e:
        logger.warning(f"Error in regexp_replace: {e}")
        return str(string) if string is not None else ""

def regexp_match(pattern: str, string: str, flags: Optional[str] = None) -> bool:
    """SQLite regexp function for pattern matching"""
    try:
        if string is None or pattern is None:
            return False
            
        string = str(string)
        pattern = str(pattern)
        
        re_flags = 0
        if flags:
            flags = str(flags).lower()
            if 'i' in flags:
                re_flags |= re.IGNORECASE
            if 'm' in flags:
                re_flags |= re.MULTILINE
            if 's' in flags:
                re_flags |= re.DOTALL
        
        return bool(re.search(pattern, string, re_flags))
        
    except Exception as e:
        logger.warning(f"Error in regexp_match: {e}")
        return False

def regexp_extract(pattern: str, string: str, group: int = 0, flags: Optional[str] = None) -> Optional[str]:
    """Extract a specific group from a regex match"""
    try:
        if string is None or pattern is None:
            return None
            
        string = str(string)
        pattern = str(pattern)
        
        re_flags = 0
        if flags:
            flags = str(flags).lower()
            if 'i' in flags:
                re_flags |= re.IGNORECASE
            if 'm' in flags:
                re_flags |= re.MULTILINE
            if 's' in flags:
                re_flags |= re.DOTALL
        
        match = re.search(pattern, string, re_flags)
        if match:
            return match.group(group)
        return None
        
    except Exception as e:
        logger.warning(f"Error in regexp_extract: {e}")
        return None





def split_string(string: str, delimiter: str = ',', index: Optional[int] = None) -> str:
    """Split string by delimiter and return specific index or all parts"""
    try:
        if string is None:
            return ""
        
        string = str(string)
        delimiter = str(delimiter) if delimiter is not None else ','
        
        parts = string.split(delimiter)
        
        if index is not None:
            try:
                return parts[int(index)] if 0 <= int(index) < len(parts) else ""
            except (ValueError, IndexError):
                return ""
        
        return delimiter.join(parts)
        
    except Exception as e:
        logger.warning(f"Error in split_string: {e}")
        return str(string) if string is not None else ""

def proper_case(string: str) -> str:
    """Convert string to proper case (Title Case)"""
    try:
        if string is None:
            return ""
        return str(string).title()
    except Exception as e:
        logger.warning(f"Error in proper_case: {e}")
        return str(string) if string is not None else ""

def reverse_string(string: str) -> str:
    """Reverse a string"""
    try:
        if string is None:
            return ""
        return str(string)[::-1]
    except Exception as e:
        logger.warning(f"Error in reverse_string: {e}")
        return str(string) if string is not None else ""

def left_pad(string: str, length: int, pad_char: str = ' ') -> str:
    """Left pad string to specified length"""
    try:
        if string is None:
            string = ""
        string = str(string)
        length = int(length)
        pad_char = str(pad_char)[0] if pad_char else ' '
        
        return string.rjust(length, pad_char)
    except Exception as e:
        logger.warning(f"Error in left_pad: {e}")
        return str(string) if string is not None else ""

def right_pad(string: str, length: int, pad_char: str = ' ') -> str:
    """Right pad string to specified length"""
    try:
        if string is None:
            string = ""
        string = str(string)
        length = int(length)
        pad_char = str(pad_char)[0] if pad_char else ' '
        
        return string.ljust(length, pad_char)
    except Exception as e:
        logger.warning(f"Error in right_pad: {e}")
        return str(string) if string is not None else ""





def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default value for division by zero"""
    try:
        num = float(numerator) if numerator is not None else 0.0
        den = float(denominator) if denominator is not None else 0.0
        default_val = float(default) if default is not None else 0.0
        
        if den == 0:
            return default_val
        return num / den
    except Exception as e:
        logger.warning(f"Error in safe_divide: {e}")
        return float(default) if default is not None else 0.0

def percentage(part: float, whole: float, decimals: int = 2) -> float:
    """Calculate percentage with specified decimal places"""
    try:
        part_val = float(part) if part is not None else 0.0
        whole_val = float(whole) if whole is not None else 0.0
        decimals = int(decimals) if decimals is not None else 2
        
        if whole_val == 0:
            return 0.0
        
        result = (part_val / whole_val) * 100
        return round(result, decimals)
    except Exception as e:
        logger.warning(f"Error in percentage: {e}")
        return 0.0

def power_of(base: float, exponent: float) -> float:
    """Calculate base raised to the power of exponent"""
    try:
        base_val = float(base) if base is not None else 0.0
        exp_val = float(exponent) if exponent is not None else 0.0
        
        return math.pow(base_val, exp_val)
    except Exception as e:
        logger.warning(f"Error in power_of: {e}")
        return 0.0





def date_add_days(date_str: str, days: int, format_str: str = '%Y-%m-%d') -> str:
    """Add days to a date string"""
    try:
        if date_str is None:
            return ""
        
        date_str = str(date_str)
        days = int(days) if days is not None else 0
        format_str = str(format_str) if format_str is not None else '%Y-%m-%d'
        
        date_obj = datetime.strptime(date_str, format_str)
        new_date = date_obj + timedelta(days=days)
        return new_date.strftime(format_str)
    except Exception as e:
        logger.warning(f"Error in date_add_days: {e}")
        return str(date_str) if date_str is not None else ""

def date_diff_days(date1: str, date2: str, format_str: str = '%Y-%m-%d') -> int:
    """Calculate difference in days between two dates"""
    try:
        if date1 is None or date2 is None:
            return 0
        
        date1 = str(date1)
        date2 = str(date2)
        format_str = str(format_str) if format_str is not None else '%Y-%m-%d'
        
        d1 = datetime.strptime(date1, format_str)
        d2 = datetime.strptime(date2, format_str)
        
        return (d2 - d1).days
    except Exception as e:
        logger.warning(f"Error in date_diff_days: {e}")
        return 0

def format_date(date_str: str, input_format: str = '%Y%m%d', output_format: str = '%Y-%m-%d') -> str:
    """Convert date from one format to another"""
    try:
        if date_str is None:
            return ""
        
        date_str = str(date_str)
        input_format = str(input_format) if input_format is not None else '%Y%m%d'
        output_format = str(output_format) if output_format is not None else '%Y-%m-%d'
        
        date_obj = datetime.strptime(date_str, input_format)
        return date_obj.strftime(output_format)
    except Exception as e:
        logger.warning(f"Error in format_date: {e}")
        return str(date_str) if date_str is not None else ""





def json_extract_value(json_str: str, key: str) -> str:
    """Extract value from JSON string by key"""
    try:
        if json_str is None or key is None:
            return ""
        
        json_str = str(json_str)
        key = str(key)
        
        data = json.loads(json_str)
        return str(data.get(key, ""))
    except Exception as e:
        logger.warning(f"Error in json_extract_value: {e}")
        return ""

def is_valid_json(json_str: str) -> bool:
    """Check if string is valid JSON"""
    try:
        if json_str is None:
            return False
        
        json_str = str(json_str)
        json.loads(json_str)
        return True
    except Exception:
        return False





def is_numeric(value: str) -> bool:
    """Check if string represents a numeric value"""
    try:
        if value is None:
            return False
        
        str_val = str(value).strip()
        if not str_val:
            return False
        
        float(str_val)
        return True
    except ValueError:
        return False

def is_email(email: str) -> bool:
    """Basic email validation"""
    try:
        if email is None:
            return False
        
        email = str(email).strip()
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    except Exception:
        return False

def is_phone(phone: str) -> bool:
    """Basic phone number validation"""
    try:
        if phone is None:
            return False
        
        phone = str(phone).strip()

        clean_phone = re.sub(r'[^\d]', '', phone)
        
        return 7 <= len(clean_phone) <= 15
    except Exception:
        return False
def to_date(date_str : str):
    return date_str[0:4] + '-' + date_str[4:6] + '-' + date_str[6:8]

def add_sqlite_functions(conn: sqlite3.Connection) -> bool:
    """
    Add all custom utility functions to a SQLite connection
    
    Args:
        conn (sqlite3.Connection): SQLite database connection
    
    Returns:
        bool: True if functions were added successfully, False otherwise
    """
    try:
        functions_added = 0
        

        conn.create_function("regexp_replace", 3, lambda p, r, s: regexp_replace(p, r, s))
        conn.create_function("regexp_replace", 4, regexp_replace)
        conn.create_function("regexp", 2, lambda p, s: regexp_match(p, s))
        conn.create_function("regexp", 3, regexp_match)
        conn.create_function("regexp_match", 2, lambda p, s: regexp_match(p, s))
        conn.create_function("regexp_match", 3, regexp_match)
        conn.create_function("regexp_extract", 2, lambda p, s: regexp_extract(p, s))
        conn.create_function("regexp_extract", 3, lambda p, s, g: regexp_extract(p, s, g))
        conn.create_function("regexp_extract", 4, regexp_extract)
        functions_added += 9
        

        conn.create_function("split_string", 2, lambda s, d: split_string(s, d))
        conn.create_function("split_string", 3, split_string)
        conn.create_function("proper_case", 1, proper_case)
        conn.create_function("reverse_string", 1, reverse_string)
        conn.create_function("left_pad", 2, lambda s, l: left_pad(s, l))
        conn.create_function("left_pad", 3, left_pad)
        conn.create_function("right_pad", 2, lambda s, l: right_pad(s, l))
        conn.create_function("right_pad", 3, right_pad)
        functions_added += 7
        

        conn.create_function("safe_divide", 2, lambda n, d: safe_divide(n, d))
        conn.create_function("safe_divide", 3, safe_divide)
        conn.create_function("percentage", 2, lambda p, w: percentage(p, w))
        conn.create_function("percentage", 3, percentage)
        conn.create_function("power_of", 2, power_of)
        functions_added += 5
        

        conn.create_function("date_add_days", 2, lambda d, n: date_add_days(d, n))
        conn.create_function("date_add_days", 3, date_add_days)
        conn.create_function("date_diff_days", 2, lambda d1, d2: date_diff_days(d1, d2))
        conn.create_function("date_diff_days", 3, date_diff_days)
        conn.create_function("format_date", 1, lambda d: format_date(d))
        conn.create_function("format_date", 2, lambda d, i: format_date(d, i))
        conn.create_function("format_date", 3, format_date)
        functions_added += 7
        

        conn.create_function("json_extract_value", 2, json_extract_value)
        conn.create_function("is_valid_json", 1, is_valid_json)
        functions_added += 2
        

        conn.create_function("is_numeric", 1, is_numeric)
        conn.create_function("is_email", 1, is_email)
        conn.create_function("is_phone", 1, is_phone)
        functions_added += 3     
        
        conn.create_function("to_date", 1, to_date)
        functions_added += 1
        return True
        
    except Exception as e:
        logger.error(f"Error adding custom functions to SQLite connection: {e}")
        return False

def get_connection_with_functions(db_path: str) -> sqlite3.Connection:
    """
    Create a SQLite connection with all custom functions pre-loaded
    
    Args:
        db_path (str): Path to the SQLite database
    
    Returns:
        sqlite3.Connection: SQLite connection with custom functions enabled
    """
    try:
        conn = sqlite3.connect(db_path)
        
        if add_sqlite_functions(conn):
            logger.info(f"Created SQLite connection with custom functions for: {db_path}")
            return conn
        else:
            logger.warning(f"Created SQLite connection but failed to add custom functions for: {db_path}")
            return conn
            
    except Exception as e:
        logger.error(f"Error creating SQLite connection: {e}")
        raise
