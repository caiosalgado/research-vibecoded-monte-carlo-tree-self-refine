#!/usr/bin/env python3
"""
Constants used across the Monte Carlo Tree Self-Refine system
Centralized location for hardcoded values
"""

# Default model configurations
DEFAULT_MODEL = "ollama:gemma3:1b"
DEFAULT_TEMPERATURE = 0.7

# System prompts
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
EXPERT_PROGRAMMER_PROMPT = "You are an expert Python programmer. Provide clean, efficient code solutions. Follow the exact format requested."

# File paths
PROBLEMS_DATA_FILE = "data/leetcode_problems.json"

# Default problem IDs for testing
DEFAULT_PROBLEM_ID = "twoSum"

# Evaluation settings
DEFAULT_TIMEOUT_SECONDS = 3.0
OLLAMA_TIMEOUT = 600

# MCTS settings
REWARD_PARSING_RETRIES = 3 