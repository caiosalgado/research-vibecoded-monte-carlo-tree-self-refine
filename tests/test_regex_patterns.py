#!/usr/bin/env python3
"""
Comprehensive tests for regex patterns
Tests all patterns with edge cases, malformed inputs, and boundary conditions
"""

import pytest
import json
import time
from typing import List, Optional, Dict, Any
from unittest.mock import MagicMock

# Import the module we're testing
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluator import CodeTester, extract_code_delimiters
from src.prompt_templates import call_reward
from src.regex_patterns import (
    # Patterns
    PATTERN_CODE_PYTHON_BLOCK,
    PATTERN_CODE_FUNCTION_DEF,
    PATTERN_SCORE_QUOTED_COLON,
    PATTERN_SCORE_UNQUOTED_ASSIGN,
    PATTERN_JSON_SCORE_BLOCK,
    PATTERN_JSON_BLOCK,
    PATTERN_JSON_QUOTE_KEY,
    PATTERN_JSON_QUOTE_VALUE,
    PATTERN_NUMBER_ANY,
    PATTERN_NUMBER_INTEGER,
    PATTERN_NUMBER_FLOAT,
    SCORE_PATTERNS,
    
    # Compiled patterns
    COMPILED_CODE_PYTHON_BLOCK,
    COMPILED_CODE_FUNCTION_DEF,
    COMPILED_SCORE_QUOTED_COLON,
    COMPILED_SCORE_UNQUOTED_ASSIGN,
    COMPILED_JSON_SCORE_BLOCK,
    COMPILED_JSON_BLOCK,
    COMPILED_JSON_QUOTE_KEY,
    COMPILED_JSON_QUOTE_VALUE,
    COMPILED_NUMBER_ANY,
    COMPILED_NUMBER_INTEGER,
    COMPILED_NUMBER_FLOAT,
    
    # Functions
    regex_extract_code_from_markdown,
    regex_extract_function_definitions,
    regex_parse_score_from_response,
    regex_extract_json_score_block,
    regex_normalize_json_quotes,
    regex_extract_score_from_json_block,
    regex_extract_all_numbers,
    regex_find_fallback_score,
    regex_parse_score_comprehensive,
)


class TestIntegrationScenarios:
    """Integration tests with real-world LLM responses"""
    
    def test_full_evaluation_flow(self):
        """Test complete flow: markdown → code extraction → evaluation"""
        # Mock problem data
        problem_data = {
            "id": "test_func",
            "title": "Test Function",
            "description": "Return input plus one",
            "function_signature": "def test_func(x: int) -> int:",
            "tests": [{"input": [1], "expected": 2}]
        }
        
        # Simulate LLM response with markdown code block
        llm_response = '''Let me solve this step by step:
        
        ```python
        def test_func(x: int) -> int:
            """Add one to input"""
            return x + 1
        ```
        
        This solution handles the case where...'''
        
        # Extract code using our regex
        code = extract_code_delimiters(llm_response)
        assert "def test_func" in code
        assert "return x + 1" in code
        
        # Verify code actually runs through evaluator
        tester = CodeTester()
        results = tester.run_evaluation(code, problem_data)
        assert results["passed"] == 1
        assert results["failed"] == 0

    def test_real_reward_responses(self):
        """Test score parsing with actual LLM reward responses"""
        test_cases = [
            # Standard JSON response following our template
            {
                "response": '''Here's my analysis:
                {
                    "analysis": "The solution is mostly correct but could be optimized. Time complexity is O(n) which meets requirements. -15 points for not handling edge cases.",
                    "score": -85
                }
                
                Let me explain further...''',
                "expected_score": -85
            },
            
            # Response with version numbers to ignore
            {
                "response": '''Analysis of solution v1.2.3:
                {
                    "analysis": "Implementation uses deprecated API from version 2.1.0",
                    "score": -90
                }''',
                "expected_score": -90
            },
            
            # Scientific notation with explicit plus
            {
                "response": '''{"analysis": "Nearly perfect!", "score": +9.5e1}''',
                "expected_score": 95.0
            },
            
            # Complex JSON with escaped quotes and single-quoted keys
            {
                "response": """{'analysis': "Complex \"quoted\" analysis", 'score': "-73"}""",
                "expected_score": -73
            },
            
            # Multiple numbers but only score matters
            {
                "response": '''Performance analysis:
                - Runtime: 45ms
                - Memory: 12MB
                - Test cases: 8/10 passed
                
                {"analysis": "Good attempt", "score": -65}''',
                "expected_score": -65
            }
        ]
        
        for case in test_cases:
            score = regex_parse_score_comprehensive(case["response"])
            assert score == case["expected_score"]

    def test_large_input_performance(self):
        """Test regex performance on large inputs"""
        # Generate large code block
        large_code = "def test():\n" + "\n".join(f"    print({i})" for i in range(1000))
        text = f"```python\n{large_code}\n```"
        
        # Measure extraction time
        start = time.time()
        result = regex_extract_code_from_markdown(text)
        duration = time.time() - start
        
        assert duration < 0.1  # Should complete within 100ms
        assert "def test()" in result
        assert "print(999)" in result

    @pytest.mark.parametrize("response,expected", [
        # Test first block is chosen
        ('''```python
        def first(): pass
        ```
        Some text
        ```python
        def second(): pass
        ```''', "def first(): pass"),
        
        # Test no language specified
        ('''```
        def test(): pass
        ```''', None),
        
        # Test with line numbers
        ('''```python
        1| def numbered(): pass
        2|     return True
        ```''', "def numbered(): pass\n    return True")
    ])
    def test_code_block_variations(self, response: str, expected: str):
        """Test various code block formats and ensure consistent behavior"""
        result = regex_extract_code_from_markdown(response)
        if expected is None:
            assert result is None
        else:
            assert result.strip() == expected.strip()


class TestCodeExtractionPatterns:
    """Test code extraction patterns and functions"""
    
    def test_extract_code_from_markdown_basic(self):
        """Test basic code extraction from markdown"""
        text = "Here's some code:\n```python\ndef hello():\n    print('world')\n```"
        result = regex_extract_code_from_markdown(text)
        assert result == "def hello():\n    print('world')"
    
    def test_extract_code_from_markdown_multiline(self):
        """Test multiline code extraction"""
        text = """
        Some explanation
        ```python
        def solution(nums):
            result = []
            for num in nums:
                if num > 0:
                    result.append(num * 2)
            return result
        ```
        More text
        """
        result = regex_extract_code_from_markdown(text)
        expected = "def solution(nums):\n            result = []\n            for num in nums:\n                if num > 0:\n                    result.append(num * 2)\n            return result"
        assert result == expected
    
    def test_extract_code_from_markdown_no_code(self):
        """Test when no code blocks exist"""
        text = "This is just regular text without any code blocks"
        result = regex_extract_code_from_markdown(text)
        assert result is None
    
    def test_extract_code_from_markdown_empty_block(self):
        """Test empty code block"""
        text = "```python\n```"
        result = regex_extract_code_from_markdown(text)
        assert result == ""
    
    def test_extract_code_from_markdown_multiple_blocks(self):
        """Test multiple code blocks - should return first one"""
        text = """
        ```python
        def first():
            pass
        ```
        Some text
        ```python
        def second():
            pass
        ```
        """
        result = regex_extract_code_from_markdown(text)
        assert "def first():" in result
        assert "def second():" not in result
    
    def test_extract_code_from_markdown_with_whitespace(self):
        """Test code block with extra whitespace"""
        text = "```python\n\n    def test():\n        return True\n\n```"
        result = regex_extract_code_from_markdown(text)
        assert result == "def test():\n        return True"
    
    def test_extract_code_from_markdown_no_language(self):
        """Test code extraction with no language specifier"""
        text = "```\ndef main():\n    print('hello')\n```"
        # Current implementation is specific to ```python, so this should fail
        # If behavior is changed, this test should be updated
        assert regex_extract_code_from_markdown(text) is None
    
    def test_extract_function_definitions_single(self):
        """Test extracting single function definition"""
        text = """
        def my_function(x, y):
            return x + y
        """
        result = regex_extract_function_definitions(text)
        assert len(result) == 1
        assert "def my_function(x, y):" in result[0]
    
    def test_extract_function_definitions_multiple(self):
        """Test extracting multiple function definitions"""
        text = """
        def func1():
            pass
        
        class MyClass:
            def method1(self):
                pass
        
        def func2(a, b, c):
            return a + b + c
        """
        result = regex_extract_function_definitions(text)
        assert len(result) >= 2
    
    def test_extract_function_definitions_complex_signatures(self):
        """Test complex function definitions (decorators, async, multiline)"""
        text = """
@decorator
async def complex_func(
    param1: str, 
    param2: int = 42
) -> bool:
    return True

def another_func(): pass
        """
        result = regex_extract_function_definitions(text)
        assert len(result) == 2
        assert "async def complex_func" in result[0]
        assert "def another_func" in result[1]
    
    def test_extract_function_definitions_indented(self):
        """Test extracting indented function definitions"""
        text = """
        if True:
            def inner_func():
                pass
        """
        result = regex_extract_function_definitions(text)
        assert len(result) == 1
    
    def test_extract_function_definitions_none(self):
        """Test when no function definitions exist"""
        text = "This is just regular code without function definitions"
        result = regex_extract_function_definitions(text)
        assert len(result) == 0


class TestScorePatterns:
    """Test score extraction patterns"""
    
    def test_score_basic_formats(self):
        """Test basic score formats with different quote styles"""
        test_cases = [
            ('{"score": -95}', -95.0),
            ("{'score': -85}", -85.0),
            ('"score\': -75', -75.0),
            ('{"score": 95}', 95.0),
            ('{"score": +95}', 95.0),
            ('{"score": -85.5}', -85.5),
            ('{"score": 0}', 0.0),
            ("score: -90", -90.0),
            ("score = -80", -80.0)
        ]
        for text, expected in test_cases:
            assert regex_parse_score_from_response(text) == expected
    
    def test_score_out_of_range(self):
        """Test scores outside valid range"""
        assert regex_parse_score_from_response('{"score": 150}') is None
        assert regex_parse_score_from_response('{"score": -150}') is None
    
    def test_score_with_custom_range(self):
        """Test score parsing with custom range"""
        text = '{"score": 200}'
        assert regex_parse_score_from_response(text, score_range=(0, 200)) == 200.0
    
    def test_score_case_insensitive(self):
        """Test case insensitive score key"""
        assert regex_parse_score_from_response('{"Score": -85}') == -85.0
        assert regex_parse_score_from_response('{"SCORE": -85}') == -85.0


class TestJSONPatterns:
    """Test JSON-specific patterns"""
    
    def test_json_score_block_variations(self):
        """Test various JSON score block formats"""
        test_cases = [
            # Basic JSON
            ('{"score": -95}', -95.0),
            # With analysis
            ('{"analysis": "test", "score": -85}', -85.0),
            # With escaped quotes
            ('{"analysis": "test \\"quoted\\" text", "score": -75}', -75.0),
            # With trailing comma
            ('{"score": -65,}', -65.0)
        ]
        for json_str, expected in test_cases:
            score = regex_extract_score_from_json_block(json_str)
            assert score == expected
    
    def test_normalize_json_quotes_complex(self):
        """Test quote normalization with complex cases"""
        test_cases = [
            # Mixed quotes
            ("{'key': 'value'}", '{"key": "value"}'),
            # Escaped quotes
            ("{'key': 'value with \"quotes\"'}", '{"key": "value with \\"quotes\\""}'),
            # Already normalized
            ('{"key": "value"}', '{"key": "value"}')
        ]
        for input_json, expected in test_cases:
            assert regex_normalize_json_quotes(input_json) == expected


class TestComprehensiveParsing:
    """Test the main comprehensive parsing function"""
    
    def test_parse_score_methods(self):
        """Test all parsing methods in order of preference"""
        test_cases = [
            # Method 1: Direct score pattern
            ('{"score": -85}', -85.0),
            # Method 2: JSON block
            ('{"analysis": "test", "score": -75}', -75.0),
            # Method 3: Fallback number (only when clearly score-related)
            ("Final score: -65", -65.0)
        ]
        for text, expected in test_cases:
            assert regex_parse_score_comprehensive(text) == expected
    
    def test_parse_score_comprehensive_empty(self):
        """Test empty/None inputs"""
        assert regex_parse_score_comprehensive("") is None
        assert regex_parse_score_comprehensive(None) is None
    
    def test_parse_score_comprehensive_no_valid_score(self):
        """Test when no valid score can be found"""
        text = "This text contains numbers like 42 and 3.14 but no clear score"
        assert regex_parse_score_comprehensive(text) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 