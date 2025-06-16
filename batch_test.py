#!/usr/bin/env python3
"""
Simple batch test script
"""

# %% Import Batch Runner
from src.batch_runner import run_batch_evaluation, generate_report, save_results

# %% Run Batch Evaluation
# Test first 3 problems
problem_ids = ["isPalindrome", "romanToInt", "twoSum"]

print("🧪 Running batch evaluation on selected problems...")
results = run_batch_evaluation(problem_ids, max_problems=3)

# %% Generate Report
generate_report(results)

# %% Save Results
save_results(results)

print("\n🎉 Batch evaluation complete!")

# %% 