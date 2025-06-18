#!/usr/bin/env python3
"""
Test script for the refinement cycle implementation
"""

from src.client import AISuiteClient
from src.mcts import MCTS
from src.constants import DEFAULT_MODEL

def test_refinement_cycle():
    """Test the refinement cycle with max_iter=1"""
    print("🧪 Testing Refinement Cycle Implementation")
    print("="*50)
    
    # Initialize LLM client
    llm = AISuiteClient(model=DEFAULT_MODEL)
    
    # Initialize MCTS with max_iter=1 for one refinement iteration
    mcts = MCTS(llm=llm, max_iter=1)
    
    # Run on twoSum problem
    problem_id = "twoSum"
    
    print(f"🎯 Running MCTS on problem: {problem_id}")
    print(f"📊 Max iterations: {mcts.max_iter}")
    print()
    
    # Run the MCTS process
    mcts.fit(problem_id)
    
    # Print final tree summary
    print("\n🌳 Final Tree Summary:")
    mcts.print_tree_summary()
    
    # Print detailed node information
    print("\n📊 Detailed Node Information:")
    for node_name in mcts.get_nodes():
        node_info = mcts.get_node_info(node_name)
        print(f"\n{node_name}:")
        for key, value in node_info.items():
            print(f"  {key}: {value}")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_refinement_cycle() 