#!/usr/bin/env python3
"""
Temporary test file for create_random_dont_know_prompt and call_reward functions
Testing with twoSum problem using random "I don't know" responses
"""

# %% Imports and Setup
from src.evaluator import get_problem, extract_code_delimiters, CodeTester
from src.client import AISuiteClient
from src.prompt_templates import create_random_dont_know_prompt, call_reward

# Initialize components
client = AISuiteClient(
    # model="ollama:gemma3:1b",
    model="ollama:qwen3:1.7b",
    # model="ollama:deepseek-r1:1.5b",
    system_prompt="You are an expert Python programmer. Provide clean, efficient code solutions. Follow the exact format requested."
)
tester = CodeTester()

# Get Problem
problem_id = "twoSum"  # Using twoSum problem
problem = get_problem(problem_id)

print("=" * 80)
print("🎯 PROBLEM SELECTED")
print("=" * 80)
print(f"ID: {problem['id']}")
print(f"Title: {problem['title']}")
print(f"Description: {problem['description']}")
print()

# Generate Random "Don't Know" Response
random_response = create_random_dont_know_prompt(problem)

print("=" * 80)
print("🎲 RANDOM 'DON'T KNOW' RESPONSE")
print("=" * 80)
print(f"Generated Response: '{random_response}'")
print()

# Since this is a "don't know" response, we'll simulate it as the LLM response
llm_response = random_response

print("=" * 80)
print("🤖 SIMULATED LLM RESPONSE")
print("=" * 80)
print(llm_response)
print()

# Extract Code (will likely be empty/None)
code = extract_code_delimiters(llm_response)

print("=" * 80)
print("🔧 EXTRACTED CODE")
print("=" * 80)
print(f"Code: {code if code else 'None - No code found in response'}")
print()

# Run Partial Evaluation
print("=" * 80)
print("⚡ RUNNING PARTIAL EVALUATION")
print("=" * 80)
partial_eval = tester.run_evaluation(code, problem, 'partial')

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

# Summary
print("=" * 80)
print("📋 EXPERIMENT SUMMARY")
print("=" * 80)
print(f"Problem: {problem['title']}")
print(f"Random Response: '{random_response}'")
print(f"Partial Accuracy: {partial_eval['accuracy']:.1%}")
print(f"Code Extracted: {'✅' if code else '❌'}")
print(f"Reward Analysis: {'✅' if reward_response else '❌'}")
print("=" * 80)
# %%