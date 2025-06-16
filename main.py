#!/usr/bin/env python3
"""
Main evaluation script following the specified pseudocode structure
"""

# %% Imports and Setup
from src.core import get_problem, create_evaluation_prompt, extract_code_delimiters, CodeTester
from aisuite_client import AISuiteClient

# Initialize components
client = AISuiteClient(
    model="ollama:gemma3:1b",
    system_prompt="You are an expert Python programmer. Provide clean, efficient code solutions. Follow the exact format requested."
)
tester = CodeTester()

# Get Problem
# problem_id = "twoSum"  # Change this to test different problems
# problem_id = "romanToInt"  # Change this to test different problems
problem_id = "addTwoNumbers"  # Change this to test different problems
problem = get_problem(problem_id)

print("=" * 80)
print("🎯 PROBLEM SELECTED")
print("=" * 80)
print(f"ID: {problem['id']}")
print(f"Title: {problem['title']}")
print(f"Description: {problem['description']}")
print()

# Create Input Prompt
input_prompt = create_evaluation_prompt(problem)

print("=" * 80)
print("📝 INPUT PROMPT")
print("=" * 80)
print(input_prompt)
print()

# LLM Response
response = client.respond(input_prompt, print_response=False)

print("=" * 80)
print("🤖 LLM OUTPUT")
print("=" * 80)
print(response)
print()

# Extract Code
code = extract_code_delimiters(response)

print("=" * 80)
print("🔧 EXTRACTED CODE")
print("=" * 80)
print(code)
print()

# Run Evaluations
print("=" * 80)
print("⚡ RUNNING EVALUATIONS")
print("=" * 80)

# Partial evaluation (visible tests only)
partial_eval = tester.run_evaluation(code, problem, 'partial')

# Full evaluation (all tests)
all_eval = tester.run_evaluation(code, problem, 'all')

print(f"Partial Tests (Visible): {partial_eval['passed']}/{partial_eval['total']} ({partial_eval['accuracy']:.1%})")
print(f"Full Tests (All): {all_eval['passed']}/{all_eval['total']} ({all_eval['accuracy']:.1%})")
print()

# Create Evaluation Results
evaluation = {
    'problem_id': problem['id'],
    'problem_title': problem['title'],
    'status': 'success' if code else 'failed',
    'code': code,
    'results': {
        'full_accuracy': all_eval['accuracy'],
        'partial_accuracy': partial_eval['accuracy'],
        'full_details': all_eval,
        'partial_details': partial_eval,
    }
}

print("=" * 80)
print("📊 FINAL EVALUATION")
print("=" * 80)
print(f"Problem: {evaluation['problem_title']}")
print(f"Status: {evaluation['status']}")
print(f"Full Accuracy: {evaluation['results']['full_accuracy']:.1%}")
print(f"Partial Accuracy: {evaluation['results']['partial_accuracy']:.1%}")

# Show errors if any
if all_eval['errors']:
    print(f"\n❌ Errors Found: {len(all_eval['errors'])}")
    for i, error in enumerate(all_eval['errors'][:3]):  # Show first 3 errors
        print(f"  {i+1}. {error.get('error_type', 'unknown')}: {error.get('error_message', 'N/A')}")

print("=" * 80)

# %%
