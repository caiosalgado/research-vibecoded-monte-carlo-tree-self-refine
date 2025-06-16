#!/usr/bin/env python3
"""
Simple benchmark runner - tests each challenge 3 times
"""

from src.batch_runner import run_simple_benchmark

if __name__ == "__main__":
    # run_simple_benchmark("ollama:gemma3:1b") 
    run_simple_benchmark("ollama:deepseek-r1:1.5b")
    run_simple_benchmark("ollama:qwen3:1.7b")