#!/usr/bin/env python3
"""
Test script to verify q_list functionality in MCTS nodes
"""

import sys
import os
import math
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.mcts import MCTS, Node, PromptType
from src.client import AISuiteClient
from src.constants import DEFAULT_MODEL, EXPERT_PROGRAMMER_PROMPT


def test_q_list_functionality():
    """Test that q_list properly stores reward values"""
    print("🧪 Testing q_list functionality...")
    
    # Create a simple LLM client
    llm = AISuiteClient(model=DEFAULT_MODEL, system_prompt=EXPERT_PROGRAMMER_PROMPT)
    
    # Create MCTS instance with debug enabled
    mcts = MCTS(llm=llm, max_iter=2, debug=True)
    
    print("\n1️⃣ Testing Node initialization...")
    # Test node creation
    test_node = Node(name="test_node", problem_id="twoSum", prompt_type=PromptType.WEAK_ANSWER)
    
    # Verify q_list is initialized
    print(f"✅ Node q_list initialized: {test_node.q_list}")
    print(f"✅ Node q_list length: {len(test_node.q_list)}")
    assert test_node.q_list == [], "q_list should be initialized as empty list"
    
    print("\n2️⃣ Testing manual reward addition...")
    # Test manual reward addition
    test_rewards = [10.5, 15.2, 8.7]
    for reward in test_rewards:
        test_node.q_list.append(reward)
    
    print(f"✅ Added rewards to q_list: {test_node.q_list}")
    assert len(test_node.q_list) == 3, "Should have 3 rewards in q_list"
    assert test_node.q_list == test_rewards, "q_list should contain the test rewards"
    
    print("\n3️⃣ Testing to_dict includes q_list...")
    # Test to_dict includes q_list
    node_dict = test_node.to_dict()
    print(f"✅ Node dict includes q_list: {'q_list' in node_dict}")
    print(f"✅ Node dict q_list: {node_dict.get('q_list')}")
    print(f"✅ Node dict q_list_length: {node_dict.get('q_list_length')}")
    
    assert 'q_list' in node_dict, "to_dict should include q_list"
    assert 'q_list_length' in node_dict, "to_dict should include q_list_length"
    assert node_dict['q_list'] == test_rewards, "q_list in dict should match node q_list"
    assert node_dict['q_list_length'] == 3, "q_list_length should be 3"
    
    print("\n✅ All q_list functionality tests passed!")
    
    return True


def test_full_mcts_with_q_list():
    """Test MCTS run to verify q_list is populated during execution"""
    print("\n🚀 Testing full MCTS run with q_list...")
    
    # Create LLM client
    llm = AISuiteClient(model=DEFAULT_MODEL, system_prompt=EXPERT_PROGRAMMER_PROMPT)
    
    # Create MCTS instance with debug enabled and 1 iteration
    mcts = MCTS(llm=llm, max_iter=1, debug=True)
    
    print("\n🎯 Running MCTS fit...")
    # Run MCTS on a simple problem
    mcts.fit("twoSum")
    
    print("\n📊 Checking q_list values in all nodes...")
    # Check that all non-root nodes have q_list populated
    for node_name, node in mcts.nodes.items():
        if node_name != "None":  # Skip root node
            print(f"Node {node_name}:")
            print(f"  - Reward: {node.reward}")
            print(f"  - q_list: {node.q_list}")
            print(f"  - q_list length: {len(node.q_list)}")
            
            # Verify q_list has at least one value
            assert len(node.q_list) >= 1, f"Node {node_name} should have at least one reward in q_list"
            
            # If it's been selected for refinement, it might have more than one reward
            if len(node.q_list) > 1:
                print(f"  - 🔄 Node was selected and recalculated {len(node.q_list)-1} times")
    
    print("\n✅ Full MCTS q_list integration test passed!")
    return True


def test_uct_calculation_with_q_list():
    """Test that UCT calculation properly uses q_list for visit count and new Q(a) formula"""
    print("\n🧮 Testing UCT calculation with q_list...")
    
    # Create a test node
    test_node = Node(name="test_uct", problem_id="twoSum", prompt_type=PromptType.WEAK_ANSWER)
    
    print("\n1️⃣ Testing UCT with empty q_list...")
    # Empty q_list should result in 0 exploitation value
    uct_value = test_node.calculate_uct()
    print(f"✅ Empty q_list UCT: {uct_value:.4f}")
    print(f"✅ Visit count (from q_list): {test_node.visit_count}")
    assert test_node.visit_count == 0, "Visit count should be 0 for empty q_list"
    
    print("\n2️⃣ Testing UCT with single reward...")
    # Add one reward
    test_node.q_list.append(15.0)
    uct_value = test_node.calculate_uct()
    print(f"✅ Single reward UCT: {uct_value:.4f}")
    print(f"✅ Visit count (from q_list): {test_node.visit_count}")
    print(f"✅ Q(a) should equal single reward: {test_node.q_list[0]}")
    assert test_node.visit_count == 1, "Visit count should be 1 for single reward"
    
    print("\n3️⃣ Testing UCT with multiple rewards...")
    # Add more rewards to test the new formula: (min + avg) / 2
    test_node.q_list.extend([10.0, 20.0, 5.0])  # Now has [15.0, 10.0, 20.0, 5.0]
    
    min_reward = min(test_node.q_list)  # 5.0
    avg_reward = sum(test_node.q_list) / len(test_node.q_list)  # (15+10+20+5)/4 = 12.5
    expected_qa = (min_reward + avg_reward) / 2  # (5.0 + 12.5) / 2 = 8.75
    
    uct_value = test_node.calculate_uct()
    print(f"✅ Multi-reward UCT: {uct_value:.4f}")
    print(f"✅ Visit count (from q_list): {test_node.visit_count}")
    print(f"✅ q_list: {test_node.q_list}")
    print(f"✅ Min reward: {min_reward}")
    print(f"✅ Avg reward: {avg_reward:.2f}")
    print(f"✅ Expected Q(a): {expected_qa:.2f}")
    
    assert test_node.visit_count == 4, "Visit count should be 4 for four rewards"
    
    # The UCT value should include the new Q(a) calculation plus exploration term
    # Since we don't have a parent, parent_visits = 1, so log(parent_visits + 1) = log(2)
    exploration_constant = 1.414
    parent_visits = 1  # No parent, so centralized method returns 1
    exploration_term = exploration_constant * math.sqrt(math.log(parent_visits + 1) / (4 + 1e-6))
    expected_uct = expected_qa + exploration_term
    
    print(f"✅ Expected total UCT: {expected_uct:.4f}")
    print(f"✅ Actual UCT: {uct_value:.4f}")
    
    # Allow small floating point differences
    assert abs(uct_value - expected_uct) < 0.01, f"UCT calculation mismatch: expected {expected_uct:.4f}, got {uct_value:.4f}"
    
    print("\n✅ All UCT calculation tests passed!")
    return True


if __name__ == "__main__":
    print("🧪 Starting q_list functionality tests...\n")
    
    try:
        # Test basic q_list functionality
        test_q_list_functionality()
        
        # Test UCT calculation with new formula
        test_uct_calculation_with_q_list()
        
        # Test full MCTS integration
        test_full_mcts_with_q_list()
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 