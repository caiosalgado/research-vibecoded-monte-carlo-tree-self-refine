#!/usr/bin/env python3
"""
Prompt Templates for Different Evaluation Strategies
Contains various prompt templates for different types of model evaluation
"""

import random


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
    
    dont_know_responses = [
        "I Don't Know",
        "I can't understand this question.",
        "I can't help with this question.",
        "I don't know how to solve this question.",
        "I don't know the answer to this question.",
        "I don't know the answer to this question, sorry."
    ]
    
    # Select a random response
    random_response = random.choice(dont_know_responses)
    
    return random_response


def create_evaluation_prompt(problem_data):
    """Create structured prompt with hidden last test case - original version"""
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

**Task:** Analyze this Answer Strictly and Critic, and point out every flaw for every possible imperfect to minus every possible score! You need to be very harsh and mean in calculating grades, and never give full marks to ensure that the marks are authoritative.

Output a score between [-100,+100], i.e. from -100 to +100.

Provide your response in the following JSON format:
{{
    'analysis': 'Your detailed analysis here',
    'score': integer_score_between_minus_100_and_100
}}

Example response:
{{
    'analysis': 'This response demonstrates major flaws in multiple areas. The code contains serious logic errors, fails to handle edge cases properly, and shows poor optimization. The test results reveal critical failures, with an unacceptably low accuracy rate. The code structure violates basic principles of clean coding, lacks proper error handling, and fails to meet the specified requirements. The implementation is inefficient and would likely fail in production scenarios.',
    'score': -85
}}
"""
    
    return prompt


