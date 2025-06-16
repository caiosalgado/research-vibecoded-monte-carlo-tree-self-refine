#!/usr/bin/env python3
"""
Usage example for AISuiteClient
"""

from aisuite_client import AISuiteClient

# Create client with default settings
client = AISuiteClient()

# Basic usage
response = client.respond("What is machine learning?")

# Use without printing
response = client.respond("Explain recursion briefly", print_response=False)
print(f"Got response: {len(response)} chars")

# Change system prompt
client.set_system_prompt("You are a poet. Respond in verse.")
client.respond("Tell me about the sun")

# Change model if you have others available
# client.set_model("ollama:gemma3:1b")
# client.respond("Hello again!") 