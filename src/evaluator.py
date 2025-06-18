#!/usr/bin/env python3
"""
Core functions for automated code evaluation
Essential components only
"""

import re
import json
import traceback
from typing import List, Optional, Any
from .client import AISuiteClient
from .constants import PROBLEMS_DATA_FILE
from multiprocessing import Process, Queue


def get_problem(problem_id, data_file=PROBLEMS_DATA_FILE):
    """Get problem data by ID"""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    for problem in data['problems']:
        if problem['id'] == problem_id:
            return problem
    
    raise ValueError(f"Problem with ID '{problem_id}' not found")


# TODO: Remove this function, since we are using the prompt template file now.
# TODO: Before removing, make sure they are the same. This version is the right one.
def create_evaluation_prompt(problem_data):
    """Create structured prompt with hidden last test case"""
    visible_tests = problem_data['tests'][:-1]  # Hide last test
    
    # Format visible test cases
    test_cases_str = ""
    for i, test in enumerate(visible_tests, 1):
        test_cases_str += f"Test {i}: Input {test['input']} → Expected Output: {test['expected']}\n"
    
    prompt = f"""Solve this coding problem:

**Problem:** {problem_data['title']}

**Description:** {problem_data['description']}

**Constraints:** {problem_data['constraints']}

**Visible Test Cases:**
{test_cases_str}
**Note:** There is 1 additional hidden test case for evaluation.

**CRITICAL FORMATTING:** Provide your solution in this EXACT format:
```python
{problem_data['function_signature']}
    # Your implementation here
    pass
```

Respond with ONLY the function code block. No explanations, no additional text."""
    
    return prompt


def extract_code_delimiters(response):
    """Extract code from structured response"""
    from .regex_patterns import regex_extract_code_from_markdown
    
    # Try regex patterns first
    code = regex_extract_code_from_markdown(response)
    if code:
        return code
    
    # Fallback: try to find function definition
    lines = response.split('\n')
    code_lines = []
    in_function = False
    
    for line in lines:
        if line.strip().startswith('def '):
            in_function = True
        if in_function:
            code_lines.append(line)
            
    return '\n'.join(code_lines) if code_lines else ""


class CodeTester:
    """Essential code testing functionality"""
    
    def __init__(self):
        # Create namespace with required types and classes
        self.namespace = {
            'List': List,
            'Optional': Optional,
            'ListNode': self.create_listnode_class()
        }
    
    def create_listnode_class(self):
        """Create ListNode class for linked list problems"""
        class ListNode:
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next
                
            def to_list(self):
                """Convert linked list to Python list for comparison"""
                result = []
                current = self
                while current:
                    result.append(current.val)
                    current = current.next
                return result
            
            def __eq__(self, other):
                """Allow direct comparison with lists"""
                if isinstance(other, list):
                    return self.to_list() == other
                return False
                
        return ListNode
    
    def list_to_linked_list(self, lst):
        """Convert Python list to linked list"""
        if not lst:
            return None
        
        ListNode = self.namespace['ListNode']
        head = ListNode(lst[0])
        current = head
        
        for val in lst[1:]:
            current.next = ListNode(val)
            current = current.next
            
        return head
    
    def prepare_inputs(self, test_input, function_id):
        """Prepare test inputs based on function requirements"""
        # Handle linked list problems
        if 'addTwoNumbers' in function_id or 'mergeKLists' in function_id:
            if function_id == 'addTwoNumbers':
                # Convert two lists to linked lists
                return [self.list_to_linked_list(test_input[0]), 
                       self.list_to_linked_list(test_input[1])]
            elif function_id == 'mergeKLists':
                # Convert list of lists to list of linked lists
                return [[self.list_to_linked_list(lst) for lst in test_input[0]]]
        
        # For most problems, return as-is
        return test_input
    
    def normalize_output(self, actual):
        """Normalize output for comparison"""
        # Handle LinkedList results
        if hasattr(actual, 'to_list'):
            return actual.to_list()
        elif hasattr(actual, 'val'):  # Single node
            return [actual.val]
        return actual
    
    def run_evaluation(self, code, problem_data, test_type='all', timeout_seconds: float = 3.0):
        """
        Run evaluation on the generated code
        
        Args:
            code: Python code string to evaluate
            problem_data: Problem data from JSON
            test_type: 'all' or 'partial' (excludes last test)
            timeout_seconds: Timeout in seconds
        
        Returns:
            Dictionary with test results
        """
        tests = problem_data['tests']
        
        # Select test subset based on type
        if test_type == 'partial':
            tests = tests[:-1]
        
        results = {
            'total': len(tests),
            'passed': 0,
            'failed': 0,
            'errors': [],
            'accuracy': 0.0,
            'test_type': test_type
        }
        
        # Safety check: Handle empty or None code
        if not code or not code.strip():
            results['failed'] = results['total']  # All tests failed due to no code
            results['errors'].append({
                'error_type': 'no_code_provided',
                'error_message': 'No code was provided or extracted from the response',
                'input': 'N/A',
                'expected': 'Valid Python function',
                'actual': 'Empty or None'
            })
            return results
        
        if not tests:
            return results
        
        try:
            result_queue = Queue()
            
            def worker():
                try:
                    # Execute the code in our prepared namespace
                    exec(code, self.namespace)
                    func = self.namespace[problem_data['id']]
                    
                    for i, test in enumerate(tests):
                        try:
                            # Prepare inputs for this specific function
                            inputs = self.prepare_inputs(test['input'], problem_data['id'])
                            expected = test['expected']
                            
                            # Call the function
                            if len(inputs) == 1:
                                actual = func(inputs[0])
                            else:
                                actual = func(*inputs)
                            
                            # Normalize output for comparison
                            actual = self.normalize_output(actual)
                            
                            # Compare results
                            if actual == expected:
                                results['passed'] += 1
                            else:
                                results['failed'] += 1
                                results['errors'].append({
                                    'test_index': i,
                                    'input': test['input'],
                                    'expected': expected,
                                    'actual': actual,
                                    'error_type': 'wrong_answer'
                                })
                                
                        except Exception as e:
                            results['failed'] += 1
                            results['errors'].append({
                                'test_index': i,
                                'input': test['input'],
                                'expected': test['expected'],
                                'error_type': 'runtime_error',
                                'error_message': str(e),
                                'traceback': traceback.format_exc()
                            })
                    
                    result_queue.put({'success': True, 'results': results})
                except Exception as e:
                    result_queue.put({'success': False, 'error': str(e)})
            
            p = Process(target=worker)
            p.start()
            p.join(timeout=timeout_seconds)
            
            if p.is_alive():
                p.terminate()
                p.join()
                # Return proper format for timeout case
                results['failed'] = results['total']  # All tests failed due to timeout
                results['errors'].append({
                    'error_type': 'timeout',
                    'error_message': f'Evaluation timed out after {timeout_seconds} seconds'
                })
                results['accuracy'] = 0.0
                return results
            
            if result_queue.empty():
                # Return proper format for empty queue case
                results['failed'] = results['total']  # All tests failed
                results['errors'].append({
                    'error_type': 'unknown_error',
                    'error_message': 'No result from evaluation process'
                })
                results['accuracy'] = 0.0
                return results
            
            result = result_queue.get()
            
            if result['success']:
                results = result['results']
                results['accuracy'] = results['passed'] / results['total'] if results['total'] > 0 else 0.0
                return results
            else:
                # Return proper format for worker error case
                results['failed'] = results['total']  # All tests failed
                results['errors'].append({
                    'error_type': 'runtime_error',
                    'error_message': result['error']
                })
                results['accuracy'] = 0.0
                return results
            
        except Exception as e:
            results['errors'].append({
                'error_type': 'compilation_error',
                'error_message': str(e),
                'traceback': traceback.format_exc()
            })
            
        return results 