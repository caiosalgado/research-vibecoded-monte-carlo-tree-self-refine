#!/usr/bin/env python3
"""
Test AISuite with a coding challenge from leetcode_problems.json
"""
# %%
from aisuite_client import AISuiteClient
import json



def create_coding_prompt(problem_data):
    """Create a detailed coding prompt from problem data"""
    
    prompt = f"""Solve this coding problem:

**Problem:** {problem_data['title']}

**Description:** {problem_data['description']}

**Constraints:** {problem_data['constraints']}

**Function Signature:**
```python
{problem_data['function_signature']}
```

**Test Cases:**
"""
    
    # Show all but last test case
    for i, test in enumerate(problem_data['tests'][:-1], 1):
        prompt += f"Test {i}: Input {test['input']} → Expected Output: {test['expected']}\n"
    
    # Add note about hidden test
    prompt += "\nPlus 1 hidden test case not shown here.\n"
    
    prompt += """
**Instructions:**
1. Provide a complete, working Python solution
2. Include comments explaining your approach
3. Optimize for both time and space complexity where possible
4. Make sure your solution handles all the test cases

Please write the complete solution:"""
    
    return prompt

# %%
# Load the problems
with open('leetcode_problems.json', 'r') as f:
    data = json.load(f)

# %%
# Select the Two Sum problem (it's a classic and good for testing)
two_sum_problem = None
for problem in data['problems']:
    if problem['id'] == 'findMedianSortedArrays':
        two_sum_problem = problem
        break


# Create AISuite client with coding-focused system prompt
client = AISuiteClient(
    system_prompt="You are an expert software engineer and coding instructor. Provide clean, efficient, well-commented Python code solutions. Explain your approach clearly and consider edge cases."
)

# Create the coding prompt
coding_prompt = create_coding_prompt(two_sum_problem)
# %%
print("=" * 60)
print("CODING CHALLENGE PROMPT:")
print("=" * 60)
print(coding_prompt[:200] + "..." if len(coding_prompt) > 200 else coding_prompt)
print("\n" + "=" * 60)
print("AI SOLUTION:")
print("=" * 60)

# Get the AI's solution
solution = client.respond(coding_prompt, print_response=True)

print("\n" + "=" * 60)
print("CHALLENGE COMPLETED!")
print("=" * 60)


# %%
