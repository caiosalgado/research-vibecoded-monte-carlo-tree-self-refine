#!/usr/bin/env python3
"""
Regex Patterns Module for Monte Carlo Tree Self-Refine
Centralized location for all regex patterns with clear naming conventions

Naming Convention:
- Raw Patterns: PATTERN_<PURPOSE>_<DESCRIPTION>
- Compiled Patterns: COMPILED_<PURPOSE>_<DESCRIPTION>  
- Functions: match_<purpose>, extract_<purpose>, parse_<purpose>
"""

import re
from typing import List, Optional, Match, Tuple, Union


# =============================================================================
# CODE EXTRACTION PATTERNS
# =============================================================================

# Pattern for extracting Python code blocks from markdown
PATTERN_CODE_PYTHON_BLOCK = r'```python\n(.*?)```'
COMPILED_CODE_PYTHON_BLOCK = re.compile(PATTERN_CODE_PYTHON_BLOCK, re.DOTALL)

# Pattern for detecting function definitions
PATTERN_CODE_FUNCTION_DEF = r'(?:@\w+\s*\n)*\s*(?:async\s+)?def\s+\w+\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:'
COMPILED_CODE_FUNCTION_DEF = re.compile(PATTERN_CODE_FUNCTION_DEF, re.MULTILINE)


# =============================================================================
# SCORE/REWARD PARSING PATTERNS
# =============================================================================

# Patterns for extracting numerical scores from LLM responses
PATTERN_SCORE_QUOTED_COLON = r'["\']score["\']:\s*["\']?([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)["\']?'  # "score": -95 or 'score': "-95"
PATTERN_SCORE_UNQUOTED_ASSIGN = r'score["\']?\s*[:=]\s*["\']?([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)["\']?'  # score: -95 or score = "-95"

# Compiled score patterns for performance
COMPILED_SCORE_QUOTED_COLON = re.compile(PATTERN_SCORE_QUOTED_COLON, re.IGNORECASE)
COMPILED_SCORE_UNQUOTED_ASSIGN = re.compile(PATTERN_SCORE_UNQUOTED_ASSIGN, re.IGNORECASE)

# All score patterns combined (for legacy compatibility)
SCORE_PATTERNS = [
    PATTERN_SCORE_QUOTED_COLON,
    PATTERN_SCORE_UNQUOTED_ASSIGN
]


# =============================================================================
# JSON PARSING PATTERNS
# =============================================================================

# Pattern for extracting JSON block containing score
PATTERN_JSON_SCORE_BLOCK = r'{\s*(?:"[^"]*"\s*:\s*"[^"]*"\s*,\s*)?["\']score["\']\s*:\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?|"[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?")[\s,}]*}'
COMPILED_JSON_SCORE_BLOCK = re.compile(PATTERN_JSON_SCORE_BLOCK, re.IGNORECASE | re.DOTALL)

# Pattern for finding JSON blocks (for quote normalization)
PATTERN_JSON_BLOCK = r'\{[^{}]*["\']score["\'][^{}]*\}'
COMPILED_JSON_BLOCK = re.compile(PATTERN_JSON_BLOCK, re.DOTALL | re.IGNORECASE)

# Patterns for normalizing JSON quotes
PATTERN_JSON_QUOTE_KEY = r"'(\w+)':"  # 'key': -> "key":
PATTERN_JSON_QUOTE_VALUE = r":\s*'([^']*)'(?=\s*[,}])"  # : 'value' -> : "value"

COMPILED_JSON_QUOTE_KEY = re.compile(PATTERN_JSON_QUOTE_KEY)
COMPILED_JSON_QUOTE_VALUE = re.compile(PATTERN_JSON_QUOTE_VALUE)


# =============================================================================
# NUMBER EXTRACTION PATTERNS
# =============================================================================

# Pattern for extracting any numeric values (integers or floats)
PATTERN_NUMBER_ANY = r'([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)'
COMPILED_NUMBER_ANY = re.compile(PATTERN_NUMBER_ANY, re.IGNORECASE)

# Pattern for extracting integers only
PATTERN_NUMBER_INTEGER = r'(-?\d+)'
COMPILED_NUMBER_INTEGER = re.compile(PATTERN_NUMBER_INTEGER)

# Pattern for extracting floats only
PATTERN_NUMBER_FLOAT = r'(-?\d+\.\d+)'
COMPILED_NUMBER_FLOAT = re.compile(PATTERN_NUMBER_FLOAT)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def regex_extract_code_from_markdown(text: str) -> Optional[str]:
    """
    Extract Python code from markdown code blocks
    
    Args:
        text: Input text containing markdown code blocks
        
    Returns:
        Extracted code or None if not found
    """
    if text is None:
        return None
        
    match = COMPILED_CODE_PYTHON_BLOCK.search(text)

    if not match:
        return None
        
    code = match.group(1)
    
    # Process each line while preserving nested blocks and indentation
    lines = []
    for line in code.split('\n'):
        # If a line seems to have a line number, remove it.
        # This pattern is designed to be specific to formats like "1 | code"
        # and preserve indentation of the actual code.
        if re.match(r'^\s*\d+\s*\|', line):
            processed_line = re.sub(r'^\s*\d+\s*\|\s?', '', line)
            lines.append(processed_line.rstrip())
        else:
            # Preserve original line with its indentation
            lines.append(line.rstrip())
    
    return '\n'.join(lines).strip()


def regex_extract_function_definitions(text: str) -> List[str]:
    """
    Extract all function definition lines from text
    
    Args:
        text: Input text
        
    Returns:
        List of complete function definition lines
    """
    if text is None:
        return []
        
    # Find all function definitions
    matches = list(COMPILED_CODE_FUNCTION_DEF.finditer(text))
    definitions = []
    
    for i, match in enumerate(matches):
        # Get the start position of this match
        start = match.start()
        
        # Look backwards for decorators
        line_start = text.rfind('\n', 0, start) + 1
        if line_start < 0:
            line_start = 0
            
        # Look forward to the next function or end of text
        if i < len(matches) - 1:
            next_def = matches[i + 1].start()
        else:
            next_def = len(text)
            
        # Extract the complete function definition including decorators
        definition = text[line_start:next_def].rstrip()
        if definition:
            definitions.append(definition)
            
    return definitions


def regex_parse_score_from_response(text: str, score_range: Tuple[float, float] = (-100, 100)) -> Optional[float]:
    """
    Parse numerical score from LLM response using multiple patterns
    
    Args:
        text: Input text from LLM response
        score_range: Valid score range as (min, max) tuple
        
    Returns:
        Parsed score or None if not found/invalid
    """
    if text is None:
        return None
        
    min_score, max_score = score_range
    
    # Try each score pattern in order
    for pattern in [COMPILED_SCORE_QUOTED_COLON, COMPILED_SCORE_UNQUOTED_ASSIGN]:
        match = pattern.search(text)
        if match:
            try:
                score = float(match.group(1))
                if min_score <= score <= max_score:
                    return score
            except (ValueError, IndexError):
                continue
    
    return None


def regex_extract_json_score_block(text: str) -> Optional[str]:
    """
    Extract JSON-like block containing score from text
    
    Args:
        text: Input text
        
    Returns:
        JSON block string or None if not found
    """
    match = COMPILED_JSON_BLOCK.search(text)
    return match.group() if match else None


def regex_normalize_json_quotes(json_str: str) -> str:
    """
    Normalize single quotes to double quotes in JSON string
    
    Args:
        json_str: JSON string with potentially mixed quotes
        
    Returns:
        Normalized JSON string with double quotes
    """
    if json_str is None:
        return ""
        
    # First normalize key quotes
    normalized = COMPILED_JSON_QUOTE_KEY.sub(r'"\1":', json_str)
    
    # Then handle values with escaped quotes
    def replace_value_quotes(match):
        value = match.group(1)
        # Escape any unescaped double quotes
        value = value.replace(r'"', r'\"')
        return f': "{value}"'
        
    normalized = re.sub(r":\s*'([^']*)'(?=\s*[,}])", replace_value_quotes, normalized)
    
    return normalized


def regex_extract_score_from_json_block(text: str, score_range: Tuple[float, float] = (-100, 100)) -> Optional[float]:
    """
    Extract score from JSON-like structure in text
    
    Args:
        text: Input text containing JSON-like structure
        score_range: Valid score range as (min, max) tuple
        
    Returns:
        Extracted score or None if not found/invalid
    """
    if text is None:
        return None
        
    min_score, max_score = score_range
    
    # First normalize quotes to make parsing easier
    text = regex_normalize_json_quotes(text)
    
    # Try to find a score in a JSON block
    match = COMPILED_JSON_SCORE_BLOCK.search(text)
    if match:
        try:
            score_str = match.group(1).strip('"\'')  # Remove quotes if present
            score = float(score_str)
            if min_score <= score <= max_score:
                return score
        except (ValueError, IndexError):
            pass
            
    # Try simpler pattern if JSON block not found
    match = re.search(r'"score"\s*:\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)', text, re.IGNORECASE)
    if match:
        try:
            score = float(match.group(1))
            if min_score <= score <= max_score:
                return score
        except (ValueError, IndexError):
            pass
    
    return None


def regex_extract_all_numbers(text: str, number_range: Optional[Tuple[float, float]] = None) -> List[float]:
    """
    Extract all numeric values from text
    
    Args:
        text: Input text
        number_range: Optional range filter as (min, max) tuple
        
    Returns:
        List of extracted numbers (within range if specified)
    """
    matches = COMPILED_NUMBER_ANY.findall(text)
    numbers = []
    
    for match in matches:
        try:
            num = float(match)
            if number_range is None or (number_range[0] <= num <= number_range[1]):
                numbers.append(num)
        except ValueError:
            continue
            
    return numbers


def regex_find_fallback_score(text: str, score_range: Tuple[float, float] = (-100, 100)) -> Optional[float]:
    """
    Find any reasonable number that could be a score as fallback
    
    Args:
        text: Input text
        score_range: Valid score range as (min, max) tuple
        
    Returns:
        First valid number found or None
    """
    if text is None:
        return None
        
    min_score, max_score = score_range
    
    # Exclude version-like patterns (e.g., v1.2.3)
    text_no_versions = re.sub(r'v\d+\.\d+\.\d+', '', text)
    
    # Look for numbers that appear after score-related words
    score_indicators = [
        r'score\s*[=:]\s*([+-]?\d+(?:\.\d+)?)',
        r'points?\s*[=:]\s*([+-]?\d+(?:\.\d+)?)',
        r'grade\s*[=:]\s*([+-]?\d+(?:\.\d+)?)',
        r'rating\s*[=:]\s*([+-]?\d+(?:\.\d+)?)'
    ]
    
    for pattern in score_indicators:
        match = re.search(pattern, text_no_versions, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                if min_score <= score <= max_score:
                    return score
            except (ValueError, IndexError):
                continue
    
    return None


# =============================================================================
# COMPREHENSIVE PARSING FUNCTION
# =============================================================================

def regex_parse_score_comprehensive(text: str, score_range: Tuple[float, float] = (-100, 100)) -> Optional[float]:
    """
    Comprehensive score parsing using all available methods
    
    This function tries multiple parsing strategies in order:
    1. Direct score patterns ("score": value)
    2. JSON block extraction with score
    3. Fallback number extraction
    
    Args:
        text: Input text from LLM response
        score_range: Valid score range as (min, max) tuple
        
    Returns:
        Parsed score or None if no valid score found
    """
    if text is None:
        return None
        
    # Method 1: Direct score patterns
    score = regex_parse_score_from_response(text, score_range)
    if score is not None:
        return score
    
    # Method 2: JSON block extraction
    score = regex_extract_score_from_json_block(text, score_range)
    if score is not None:
        return score
    
    # Method 3: Fallback number extraction
    score = regex_find_fallback_score(text, score_range)
    if score is not None:
        return score
    
    return None 