#!/usr/bin/env python3
"""
Temporary test file for create_weak_answer and call_reward functions
Testing with twoSum problem
"""

# %% Imports and Setup
from src.evaluator import get_problem, extract_code_delimiters, CodeTester
from src.client import AISuiteClient
from src.prompt_templates import create_weak_answer, call_reward
from src.constants import EXPERT_PROGRAMMER_PROMPT, DEFAULT_PROBLEM_ID

# Initialize components
client = AISuiteClient(
    # model="ollama:gemma3:1b",
    model="ollama:qwen3:1.7b",
    # model="ollama:deepseek-r1:1.5b",
    system_prompt=EXPERT_PROGRAMMER_PROMPT
)
tester = CodeTester()

# Get Problem
problem_id = DEFAULT_PROBLEM_ID  # Using twoSum problem
problem = get_problem(problem_id)

print("=" * 80)
print("🎯 PROBLEM SELECTED")
print("=" * 80)
print(f"ID: {problem['id']}")
print(f"Title: {problem['title']}")
print(f"Description: {problem['description']}")
print()

# Create Weak Answer Prompt
weak_prompt = create_weak_answer(problem)

print("=" * 80)
print("📝 WEAK ANSWER PROMPT")
print("=" * 80)
print(weak_prompt)
print()

# Get LLM Response
llm_response = client.respond(weak_prompt, print_response=False)

print("=" * 80)
print("🤖 LLM RESPONSE")
print("=" * 80)
print(llm_response)
print()

# Extract Code
code = extract_code_delimiters(llm_response)

print("=" * 80)
print("🔧 EXTRACTED CODE")
print("=" * 80)
print(code)
print()

# Run Partial Evaluation
print("=" * 80)
print("⚡ RUNNING PARTIAL EVALUATION")
print("=" * 80)

# Partial evaluation (visible tests only)
partial_eval = tester.run_evaluation(code, problem, 'partial')

print(f"Partial Tests (Visible): {partial_eval['passed']}/{partial_eval['total']} ({partial_eval['accuracy']:.1%})")
print(f"Errors: {len(partial_eval['errors'])}")

if partial_eval['errors']:
    print("\nError Details:")
    for i, error in enumerate(partial_eval['errors'][:2], 1):
        print(f"  {i}. {error.get('error_type', 'unknown')}: {error}")

print()

# Create Reward Analysis Prompt
print("=" * 80)
print("📊 CREATING REWARD ANALYSIS")
print("=" * 80)
# %%
reward_prompt = call_reward(llm_response, partial_eval, problem)

print("Reward Analysis Prompt:")
print("-" * 40)
print(reward_prompt)
print()

# Get Reward Analysis
print("=" * 80)
print("🎯 REWARD ANALYSIS RESPONSE")
print("=" * 80)

reward_response = client.respond(reward_prompt, print_response=False)
print(reward_response)
print()

# %% Summary
print("=" * 80)
print("📋 EXPERIMENT SUMMARY")
print("=" * 80)
print(f"Problem: {problem['title']}")
print(f"Partial Accuracy: {partial_eval['accuracy']:.1%}")
print(f"Code Extracted: {'✅' if code else '❌'}")
print(f"Reward Analysis: {'✅' if reward_response else '❌'}")
print("=" * 80) 
# %%
