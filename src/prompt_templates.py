#!/usr/bin/env python3
"""
Prompt Templates for Different Evaluation Strategies
Contains various prompt templates for different types of model evaluation
"""

import random

from .constants import DONT_KNOW_RESPONSES


def create_weak_answer(problem_data):
    """Create structured prompt with reasoning process and verification sections"""
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

STRICT OUTPUT FORMAT - DO NOT CHANGE THIS FORMAT:
1. Your entire reply must contain **EXACTLY** three sections, in this order:
    - [reasoning process]
    - [verification]
    - [code]

2. Template (copy this and replace only the inner content):

[reasoning process]
... your reasoning process here ...

[verification]
... your verification here ...

[code]
```python
{problem_data['function_signature']}
    # Your implementation here
    pass
```

3. Addtional rules:

- Do **NOT** add text outside these three sections.
- Place the `[code]` tag directly above the opening ```python and close the block with ``` (nothing after it).
- Keep the function signature **EXACTLY** as it is.

Let's think step by step.
"""
    
    return prompt


def create_random_dont_know_prompt(problem_data):
    """Generate a random 'I don't know' response instead of solving the problem"""
    
    # Select a random response from constants
    random_response = random.choice(DONT_KNOW_RESPONSES)
    
    return random_response



def call_reward(llm_answer, partial_accuracy_feedback, problem_data):
    """
    Combine LLM answer with partial accuracy feedback to return a single reward value
    
    Args:
        llm_answer (str): The answer produced by the LLM
        partial_accuracy_feedback (dict): Feedback from partial-accuracy tests
        
    Returns:
        str: Prompt for strict analysis and scoring
    """
    
    # Format the partial accuracy feedback
    feedback_str = ""
    if partial_accuracy_feedback:
        accuracy = partial_accuracy_feedback.get('accuracy', 0)
        passed = partial_accuracy_feedback.get('passed', 0)
        total = partial_accuracy_feedback.get('total', 0)
        errors = partial_accuracy_feedback.get('errors', [])
        
        feedback_str = f"""
**Partial Accuracy Results:**
- Tests Passed: {passed}/{total} ({accuracy:.1%})
- Accuracy Score: {accuracy:.3f}
"""
        
        if errors:
            feedback_str += "\n**Errors Found:**\n"
            for i, error in enumerate(errors[:3], 1):  # Show first 3 errors
                error_type = error.get('error_type', 'unknown')
                error_msg = error.get('error_message', 'N/A')
                expected = error.get('expected', 'N/A')
                actual = error.get('actual', 'N/A')
                feedback_str += f"{i}. {error_type}: Expected {expected}, Got {actual}\n"
    
    prompt = f"""Analyze the following LLM answer and test results:

**Problem:** {problem_data['title']}

**Description:** {problem_data['description']}

**Constraints:** {problem_data['constraints']}

**LLM Answer:**
{llm_answer}

{feedback_str}

**Task:** Analyze this Answer Strictly and Critic, and point out every flaw for every possible imperfect to minus every possible score! You need to be very harsh in calculating grades, and never give full marks to ensure that the marks are authoritative.

Output a score between [-100,+100], i.e. from -100 to +100.

Provide your response in the following JSON format:
{{
    "analysis": "Your detailed analysis here",
    "score": integer_score_between_minus_100_and_100
}}

Example response 1:
{{
    "analysis": "This response ... (your analysis here)",
    "score": -85
}}

Example response 2:
{{
    "analysis": "This response ... (your analysis here)",
    "score": 85
}}
"""
    
    return prompt


def create_reflection_prompt(problem_data, llm_answer, test_performance):
    """
    Create a reflection prompt for critical analysis of the current answer
    
    Args:
        problem_data: Problem data dictionary
        llm_answer: The LLM's current answer
        test_performance: Test performance results
        
    Returns:
        str: Reflection prompt for self-criticism
    """
    # Format test performance
    performance_str = ""
    if test_performance:
        accuracy = test_performance.get('accuracy', 0)
        passed = test_performance.get('passed', 0)
        total = test_performance.get('total', 0)
        errors = test_performance.get('errors', [])
        
        performance_str = f"""
**Test Performance:**
- Tests Passed: {passed}/{total} ({accuracy:.1%})
- Accuracy Score: {accuracy:.3f}
"""
        
        if errors:
            performance_str += "\n**Errors Found:**\n"
            for i, error in enumerate(errors[:3], 1):  # Show first 3 errors
                error_type = error.get('error_type', 'unknown')
                expected = error.get('expected', 'N/A')
                actual = error.get('actual', 'N/A')
                performance_str += f"{i}. {error_type}: Expected {expected}, Got {actual}\n"
    
    prompt = f"""Analyze the following coding solution and provide critical feedback:

**Problem:** {problem_data['title']}

**Description:** {problem_data['description']}

**Constraints:** {problem_data['constraints']}

**Current Answer:**
{llm_answer}

{performance_str}

**Task:** Provide me with a reflection or feedback to correct this answer better. Analyze this answer strictly and critically. Point out every flaw and every possible imperfection to subtract every possible score. Let's think step by step.

Provide detailed, actionable feedback that can be used to improve the solution.
"""
    
    return prompt


def create_improvement_prompt(problem_data, original_answer, test_performance, reflection):
    """
    Create an improvement prompt that uses reflection to generate better code
    
    Args:
        problem_data: Problem data dictionary
        original_answer: The original LLM answer
        test_performance: Test performance results
        reflection: Critical reflection from the previous step
        
    Returns:
        str: Improvement prompt for generating better code
    """
    visible_tests = problem_data['tests'][:-1]  # Hide last test
    
    # Format visible test cases
    test_cases_str = ""
    for i, test in enumerate(visible_tests, 1):
        test_cases_str += f"Test {i}: Input {test['input']} → Expected Output: {test['expected']}\n"
    
    # Format test performance
    performance_str = ""
    if test_performance:
        accuracy = test_performance.get('accuracy', 0)
        passed = test_performance.get('passed', 0)
        total = test_performance.get('total', 0)
        errors = test_performance.get('errors', [])
        
        performance_str = f"""
**Previous Performance:**
- Tests Passed: {passed}/{total} ({accuracy:.1%})
- Accuracy Score: {accuracy:.3f}
"""
        
        if errors:
            performance_str += "\n**Previous Errors:**\n"
            for i, error in enumerate(errors[:3], 1):  # Show first 3 errors
                error_type = error.get('error_type', 'unknown')
                expected = error.get('expected', 'N/A')
                actual = error.get('actual', 'N/A')
                performance_str += f"{i}. {error_type}: Expected {expected}, Got {actual}\n"
    
    prompt = f"""Improve the following coding solution based on critical feedback:

**Problem:** {problem_data['title']}

**Description:** {problem_data['description']}

**Constraints:** {problem_data['constraints']}

**Visible Test Cases:**
{test_cases_str}
**Note:** There is 1 additional hidden test case for evaluation.

**Previous Answer:**
{original_answer}

{performance_str}

**Critical Reflection:**
{reflection}

**Task:** Using the reflection above, create an improved solution that addresses all the identified issues. Think step by step and ensure your solution handles all edge cases properly.

STRICT OUTPUT FORMAT - DO NOT CHANGE THIS FORMAT:
1. Your entire reply must contain **EXACTLY** three sections, in this order:
    - [reasoning process]
    - [verification]
    - [code]

2. Template (copy this and replace only the inner content):

[reasoning process]
... your reasoning process here, incorporating the feedback ...

[verification]
... your verification here, addressing the previous issues ...

[code]
```python
{problem_data['function_signature']}
    # Your improved implementation here
    pass
```

3. Additional rules:

- Do **NOT** add text outside these three sections.
- Place the `[code]` tag directly above the opening ```python and close the block with ``` (nothing after it).
- Keep the function signature **EXACTLY** as it is.
- Address all issues mentioned in the reflection.

Let's think step by step.
"""
    
    return prompt


