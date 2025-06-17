#!/usr/bin/env python3
"""
Example usage of MCTS as specified by the user
"""
# %%
from src.mcts import MCTS
from src.client import AISuiteClient
from src.evaluator import get_problem
from src.constants import DEFAULT_MODEL, EXPERT_PROGRAMMER_PROMPT

# Setup
llm = AISuiteClient(
    model=DEFAULT_MODEL, 
    system_prompt=EXPERT_PROGRAMMER_PROMPT
)

# Create and run MCTS (now takes problem_id instead of problem data)
mcts = MCTS(llm=llm, max_iter=0)
mcts.fit("twoSum")  # Pass problem ID, not full data
# %%
# Access functionality as user requested
nodes = mcts.get_nodes()
print("Available nodes:", nodes)
# %%
# Get conversation history for a specific node
history = mcts.get_history("root.2.weak_answer")
print(f"\nConversation history length: {len(history)}")
# %%
# Show improved node information
node_info = mcts.get_node_info("root.2.weak_answer")
print(f"\nNode info:")
print(f"  Problem ID: {node_info['problem_id']}")
print(f"  Prompt Type: {node_info['prompt_type']}")
print(f"  Code Length: {len(node_info['code'])} chars")
print(f"  Reward: {node_info['reward']}")

# %%
mcts.print_tree_summary()
# %%

# You can also access other nodes:
# history = mcts.get_history("root.1.dont_know")
# history = mcts.get_history("None")  # root node 
# %%
