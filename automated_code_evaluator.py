#!/usr/bin/env python3
"""
Automated Code Evaluator - Hybrid Approach
Combines structured prompts with comprehensive testing framework
"""

import re
import json
import traceback
from typing import List, Optional
from aisuite_client import AISuiteClient


class AutomatedTester:
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
    
    def run_evaluation(self, code, problem_data, test_type='all'):
        """
        Run evaluation on the generated code
        
        Args:
            code: Python code string to evaluate
            problem_data: Problem data from JSON
            test_type: 'all', 'partial' (excludes last test), or 'hidden' (only last test)
        
        Returns:
            Dictionary with test results
        """
        tests = problem_data['tests']
        
        # Select test subset based on type
        if test_type == 'partial':
            tests = tests[:-1]
        elif test_type == 'hidden':
            tests = tests[-1:]
        
        results = {
            'total': len(tests),
            'passed': 0,
            'failed': 0,
            'errors': [],
            'accuracy': 0.0,
            'test_type': test_type
        }
        
        if not tests:
            return results
        
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
                    
        except Exception as e:
            results['errors'].append({
                'error_type': 'compilation_error',
                'error_message': str(e),
                'traceback': traceback.format_exc()
            })
            
        results['accuracy'] = results['passed'] / results['total'] if results['total'] > 0 else 0.0
        return results


class CodeEvaluator:
    def __init__(self, model="ollama:gemma3:1b"):
        self.client = AISuiteClient(
            model=model,
            system_prompt="You are an expert Python programmer. Provide clean, efficient code solutions. Follow the exact format requested."
        )
        self.tester = AutomatedTester()
    
    def create_evaluation_prompt(self, problem_data):
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
    
    def extract_code_delimiters(self, response):
        """Extract code from structured response"""
        pattern = r'```python\s*(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        
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
    
    def evaluate_problem(self, problem_data, print_details=True):
        """Complete evaluation pipeline for a single problem"""
        if print_details:
            print(f"\n{'='*60}")
            print(f"EVALUATING: {problem_data['title']}")
            print(f"{'='*60}")
        
        # Create prompt and get response
        prompt = self.create_evaluation_prompt(problem_data)
        
        if print_details:
            print("🤖 Generating solution...")
            
        response = self.client.respond(prompt, print_response=False)
        code = self.extract_code_delimiters(response)
        
        if not code:
            return {
                'problem_id': problem_data['id'],
                'status': 'failed',
                'error': 'No code extracted from response',
                'response': response
            }
        
        if print_details:
            print("📝 Extracted Code:")
            print("-" * 40)
            print(code)
            print("-" * 40)
        
        # Run evaluations
        try:
            full_results = self.tester.run_evaluation(code, problem_data, 'all')
            partial_results = self.tester.run_evaluation(code, problem_data, 'partial')
            
            evaluation = {
                'problem_id': problem_data['id'],
                'problem_title': problem_data['title'],
                'status': 'success',
                'code': code,
                'results': {
                    'full_accuracy': full_results['accuracy'],
                    'partial_accuracy': partial_results['accuracy'],
                    'full_details': full_results,
                    'partial_details': partial_results,
                }
            }
            
            if print_details:
                print(f"\n📊 RESULTS:")
                print(f"Full Tests (All): {full_results['passed']}/{full_results['total']} ({full_results['accuracy']:.1%})")
                print(f"Partial Tests (Visible): {partial_results['passed']}/{partial_results['total']} ({partial_results['accuracy']:.1%})")
                
                if full_results['errors']:
                    print(f"\n❌ Errors found: {len(full_results['errors'])}")
                    for error in full_results['errors'][:2]:  # Show first 2 errors
                        print(f"  - {error.get('error_type', 'unknown')}: {error.get('error_message', 'N/A')}")
            
            return evaluation
            
        except Exception as e:
            return {
                'problem_id': problem_data['id'],
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'code': code
            }


def main():
    """Test the evaluator with a sample problem"""
    # Load problems
    with open('leetcode_problems.json', 'r') as f:
        data = json.load(f)
    
    evaluator = CodeEvaluator()
    
    # Test with Two Sum problem
    two_sum = next(p for p in data['problems'] if p['id'] == 'twoSum')
    result = evaluator.evaluate_problem(two_sum)
    
    print(f"\n🎯 FINAL EVALUATION:")
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        results = result['results']
        print(f"Full Accuracy: {results['full_accuracy']:.1%}")
        print(f"Partial Accuracy: {results['partial_accuracy']:.1%}")  
        print(f"Generalization (Hidden): {results['hidden_accuracy']:.1%}")


if __name__ == "__main__":
    main() 