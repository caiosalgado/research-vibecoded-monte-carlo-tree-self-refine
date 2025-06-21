#!/usr/bin/env python3
"""
Constants used across the Monte Carlo Tree Self-Refine system
Centralized location for hardcoded values
"""

# Default model configurations
DEFAULT_MODEL = "ollama:gemma3:1b"
# DEFAULT_MODEL = "ollama:qwen3:1.7b"
# DEFAULT_MODEL = "ollama:deepseek-r1:1.5b"
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
MAX_CHILDREN_PER_NODE = 3

# UCT (Upper Confidence Bound applied to Trees) settings
UCT_EXPLORATION_CONSTANT = 1.414  # sqrt(2) - common default value
UCT_EPSILON = 1e-6  # Small constant to avoid division by zero 

# Don't know response templates
DONT_KNOW_RESPONSES = [
    "I Don't Know",
    "I can't understand this question.",
    "I can't help with this question.",
    "I don't know how to solve this question.",
    "I don't know the answer to this question.",
    "I don't know the answer to this question, sorry."
]