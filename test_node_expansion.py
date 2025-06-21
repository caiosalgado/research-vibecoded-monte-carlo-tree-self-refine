#!/usr/bin/env python3
"""
Test script for node expansion logic based on node_expansion_cases.json
Validates the _can_refine_node method against predefined test cases
"""

import json
from src.client import AISuiteClient
from src.mcts import MCTS, Node, PromptType
from src.constants import DEFAULT_MODEL, MAX_CHILDREN_PER_NODE
from src.debug_utils import debug_printer

def load_test_cases():
    """Load test cases from node_expansion_cases.json"""
    with open("node_expansion_cases.json", "r") as f:
        return json.load(f)

def create_test_node(parent_reward: float, child_rewards: list) -> Node:
    """Create a test node with specified parent and child rewards"""
    # Create parent node
    parent = Node(
        name="test_parent",
        problem_id="twoSum",
        prompt_type=PromptType.WEAK_ANSWER,
        parent=None
    )
    parent.q_list = [parent_reward]  # This sets both reward and visit_count via properties
    
    # Create child nodes
    for i, child_reward in enumerate(child_rewards):
        child = Node(
            name=f"test_child_{i+1}",
            problem_id="twoSum", 
            prompt_type=PromptType.REFINEMENT,
            parent=parent
        )
        child.q_list = [child_reward]  # This sets both reward and visit_count via properties
        parent.add_child(child)
    
    return parent

def test_node_expansion(verbose: bool = True):
    """Run all node expansion test cases from JSON file"""
    print("🧪 Testing Node Expansion Logic")
    print("=" * 80)
    
    # Initialize MCTS for testing
    llm = AISuiteClient(model=DEFAULT_MODEL)
    mcts = MCTS(llm=llm, max_iter=1, debug=False)
    
    # Load test cases from JSON
    test_cases = load_test_cases()
    
    passed = 0
    failed = 0
    failed_cases = []
    
    for case in test_cases:
        case_num = case["Case"]
        num_children = case["NumChildren"]
        parent_reward = case["ParentReward"]
        child_rewards = case["ChildRewards"]
        expected_can_expand = case["CanExpand"]
        reason = case["Reason"]
        condition1 = case.get("Condition1_len_lt_max", len(child_rewards) < MAX_CHILDREN_PER_NODE)
        condition2 = case.get("Condition2_best_lt_parent", True)
        best_child = case.get("BestChild", max(child_rewards) if child_rewards else None)
        
        if verbose:
            print(f"\n📋 Case {case_num}: {num_children} children, parent reward: {parent_reward}")
            print(f"   Child rewards: {child_rewards}")
            print(f"   Expected: {'✅ Can expand' if expected_can_expand else '❌ Cannot expand'}")
            print(f"   Reason: {reason}")
        
        # Create test node
        test_node = create_test_node(parent_reward, child_rewards)
        
        # Test the expansion logic
        actual_can_expand = mcts._can_refine_node(test_node)
        
        # Verify result
        if actual_can_expand == expected_can_expand:
            if verbose:
                print(f"   🎯 PASS: Got {'✅ Can expand' if actual_can_expand else '❌ Cannot expand'}")
            passed += 1
        else:
            if verbose:
                print(f"   💥 FAIL: Expected {'✅ Can expand' if expected_can_expand else '❌ Cannot expand'}, got {'✅ Can expand' if actual_can_expand else '❌ Cannot expand'}")
            failed += 1
            failed_cases.append(case_num)
            
        # Additional debugging info from JSON
        if verbose:
            print(f"   📊 Analysis:")
            print(f"      - Condition 1 (< {MAX_CHILDREN_PER_NODE} children): {condition1}")
            print(f"      - Condition 2 (no child > parent): {condition2}")
            print(f"      - Best child reward: {best_child}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"📁 Total test cases loaded from JSON: {len(test_cases)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    if failed_cases:
        print(f"🔴 Failed cases: {failed_cases}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Node expansion logic is working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the expansion logic.")
    
    return passed, failed, failed_cases

def run_detailed_analysis():
    """Run detailed analysis of specific test cases from JSON"""
    print("\n" + "=" * 80)
    print("🔍 DETAILED ANALYSIS OF JSON TEST CASES")
    print("=" * 80)
    
    # Load test cases
    test_cases = load_test_cases()
    
    # Initialize MCTS with debug enabled for detailed output
    llm = AISuiteClient(model=DEFAULT_MODEL)
    mcts = MCTS(llm=llm, max_iter=1, debug=True)
    
    # Find interesting edge cases from JSON to analyze
    edge_cases = []
    for case in test_cases:
        num_children = case["NumChildren"]
        # Look for edge cases: at max children, over max children, etc.
        if num_children == MAX_CHILDREN_PER_NODE or num_children > MAX_CHILDREN_PER_NODE:
            edge_cases.append(case)
    
    print(f"📋 Found {len(edge_cases)} edge cases to analyze in detail:")
    
    for case in edge_cases:
        case_num = case["Case"]
        parent_reward = case["ParentReward"]
        child_rewards = case["ChildRewards"]
        expected = case["CanExpand"]
        reason = case["Reason"]
        
        print(f"\n🔬 Case {case_num}: {reason}")
        print(f"   Parent reward: {parent_reward}, Child rewards: {child_rewards}")
        
        # Create and test the node
        test_node = create_test_node(parent_reward, child_rewards)
        result = mcts._can_refine_node(test_node)
        
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"   {status}: Expected {'Can expand' if expected else 'Cannot expand'}, Got {'Can expand' if result else 'Cannot expand'}")

if __name__ == "__main__":
    # Run main test suite from JSON
    passed, failed, failed_cases = test_node_expansion(verbose=True)
    
    # Run detailed analysis of edge cases from JSON
    run_detailed_analysis()
    
    print("\n" + "=" * 80)
    print("🏁 TESTING COMPLETE")
    print("=" * 80)
    
    # Exit with appropriate code
    exit(0 if failed == 0 else 1) 