#!/usr/bin/env python3
"""
Monte Carlo Tree Search with Self-Refine
Implementation of MCTS for code generation with LLM self-refinement
"""

import json
import re
from enum import Enum
from typing import Dict, List, Optional, Any
from .client import AISuiteClient
from .evaluator import CodeTester, extract_code_delimiters, get_problem
from .prompt_templates import create_random_dont_know_prompt, create_weak_answer, call_reward
from .constants import REWARD_PARSING_RETRIES


class PromptType(Enum):
    """Enum for prompt types to avoid hardcoded strings"""
    ROOT = "root"
    RANDOM_DONT_KNOW = "random_dont_know"
    WEAK_ANSWER = "weak_answer"


def generate_prompt(problem_data: Dict[str, Any], prompt_type: PromptType) -> str:
    """
    Centralized prompt generation based on prompt type
    
    Args:
        problem_data: Problem data dictionary
        prompt_type: Type of prompt to generate
        
    Returns:
        Generated prompt string
    """
    if prompt_type == PromptType.RANDOM_DONT_KNOW:
        return create_random_dont_know_prompt(problem_data)
    elif prompt_type == PromptType.WEAK_ANSWER:
        return create_weak_answer(problem_data)
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")


class Node:
    """Represents a node in the MCTS tree - stores minimal state"""
    
    def __init__(self, name: str, problem_id: str, prompt_type: PromptType = PromptType.ROOT, parent: Optional['Node'] = None):
        self.name = name
        self.problem_id = problem_id  # Store only ID, not full data
        self.prompt_type = prompt_type
        self.parent = parent
        self.children: List['Node'] = []
        self.response: str = ""
        self.reward: float = 0.0
        self.conversation_history: List[Dict[str, str]] = []
        self.reward_history: List[Dict[str, str]] = []  # Separate reward conversations
        self.evaluation_results: Optional[Dict[str, Any]] = None  # Store evaluation results
        
    @property
    def problem_data(self) -> Dict[str, Any]:
        """Get fresh problem data from single source of truth"""
        return get_problem(self.problem_id)
    
    @property
    def code(self) -> str:
        """Extract code on demand - never stored"""
        return extract_code_delimiters(self.response)
        
    def add_child(self, child: 'Node'):
        """Add a child node"""
        child.parent = self
        self.children.append(child)
        
    def to_dict(self):
        """Convert node to dictionary for inspection"""
        return {
            'name': self.name,
            'problem_id': self.problem_id,
            'prompt_type': self.prompt_type.value,
            'reward': self.reward,
            'response_length': len(self.response),
            'code_length': len(self.code),  # Uses @property
            'children_count': len(self.children),
            'parent_name': self.parent.name if self.parent else None
        }


class MCTS:
    """Monte Carlo Tree Search with Self-Refine"""
    
    def __init__(self, llm: AISuiteClient, max_iter: int = 0):
        """
        Initialize MCTS
        
        Args:
            llm: AISuiteClient instance for LLM interactions
            max_iter: Maximum iterations (0 for initial step-by-step implementation)
        """
        self.llm = llm
        self.max_iter = max_iter
        self.root: Optional[Node] = None
        self.nodes: Dict[str, Node] = {}
        self.problem_id: Optional[str] = None  # Store only ID, not data
        self.tester = CodeTester()
        
    def fit(self, problem_id: str):
        """
        Main method to run MCTS on a problem
        
        Args:
            problem_id: Problem ID to solve (not the full data)
        """
        print("🌳 Starting MCTS fit process...")
        
        # Store problem ID only
        self.problem_id = problem_id
        
        # Step 1: Create root node
        self._create_root_node()
        
        # Step 2: Generate two children
        self._generate_initial_children()
        
        # Step 3: Evaluate children
        self._evaluate_children()
        
        print("✅ MCTS fit process completed!")
        
    def _create_root_node(self):
        """Create the root node called 'None'"""
        print("📍 Creating root node...")
        
        self.root = Node(name="None", problem_id=self.problem_id, prompt_type=PromptType.ROOT)
        self.nodes["None"] = self.root
        
        print(f"   ✓ Root node '{self.root.name}' created")
        
    def _generate_initial_children(self):
        """Generate two initial children with different prompt strategies"""
        print("👶 Generating initial children...")
        
        # Child 1: Random "don't know" response
        child1 = self._create_child("root.1.dont_know", PromptType.RANDOM_DONT_KNOW)
        
        # Child 2: Weak answer
        child2 = self._create_child("root.2.weak_answer", PromptType.WEAK_ANSWER)
        
        print(f"   ✓ Child 1: {child1.name} (Response: {len(child1.response)} chars)")
        print(f"   ✓ Child 2: {child2.name} (Response: {len(child2.response)} chars)")
    
    def _create_child(self, name: str, prompt_type: PromptType) -> Node:
        """Create a child node with given prompt type"""
        print(f"   🧠 Generating response for {name}...")
        
        # Create node
        child = Node(name=name, problem_id=self.problem_id, prompt_type=prompt_type, parent=self.root)
        
        # Generate response based on prompt type
        if prompt_type == PromptType.RANDOM_DONT_KNOW:
            # For "don't know" responses, the prompt IS the response
            child.response = generate_prompt(child.problem_data, prompt_type)
            child.conversation_history = [
                {"role": "system", "content": self.llm.system_prompt},
                {"role": "assistant", "content": child.response}
            ]
        else:
            # For other prompt types, generate prompt and get LLM response
            prompt = generate_prompt(child.problem_data, prompt_type)
            child.response = self.llm.respond(prompt, print_response=False)
            child.conversation_history = [
                {"role": "system", "content": self.llm.system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": child.response}
            ]
        
        # Add to tree structure
        self.root.add_child(child)
        self.nodes[name] = child
        
        return child
        
    def _evaluate_children(self):
        """Run reward function for each child"""
        print("🏆 Evaluating children with reward function...")
        
        for child in self.root.children:
            print(f"   📊 Evaluating {child.name}...")
            
            # Run code evaluation first (using child.code @property)
            evaluation_results = self.tester.run_evaluation(
                child.code,  # Uses @property to extract code on demand
                child.problem_data,  # Uses @property to get fresh problem data
                test_type='partial'
            )
            
            # Store evaluation results in node
            child.evaluation_results = evaluation_results
            
            # Create reward prompt using the evaluation results
            reward_prompt = call_reward(
                child.response, 
                evaluation_results, 
                child.problem_data  # Uses @property
            )
            
            # Parse reward from response with retry mechanism
            child.reward = self._parse_reward_with_retry(reward_prompt, child)
            
            print(f"     ✓ Reward: {child.reward}")
            print(f"     ✓ Code tests: {evaluation_results['passed']}/{evaluation_results['total']} passed")
    
    def _parse_reward_with_retry(self, reward_prompt: str, child: Node) -> float:
        """Parse reward score from LLM response with retry mechanism"""
        for attempt in range(REWARD_PARSING_RETRIES):
            # Get reward from LLM  
            reward_response = self.llm.respond(reward_prompt, print_response=False)
            
            # Store in reward history (separate from conversation history)
            child.reward_history.append({
                "attempt": attempt + 1,
                "role": "user", 
                "content": reward_prompt
            })
            child.reward_history.append({
                "attempt": attempt + 1,
                "role": "assistant", 
                "content": reward_response
            })
            
            # Try to parse the reward
            parsed_reward = self._parse_reward(reward_response)
            if parsed_reward is not None:  # Successfully parsed
                print(f"     ✓ Reward parsed successfully on attempt {attempt + 1}")
                return parsed_reward
            else:
                print(f"     ⚠️  Parse attempt {attempt + 1} failed, retrying...")
        
        # All attempts failed
        print(f"     ❌ All {REWARD_PARSING_RETRIES} attempts failed, setting reward to -101")
        return -101.0
    
    def _parse_reward(self, reward_response: str) -> Optional[float]:
        """Parse reward score from LLM response - returns None if parsing fails"""
        try:
            # First try to find JSON block
            json_match = re.search(r'\{[^{}]*["\']score["\'][^{}]*\}', reward_response, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_str = json_match.group()
                # Handle single quotes in JSON by replacing with double quotes for keys
                json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
                reward_data = json.loads(json_str)
                score = reward_data.get('score')
                if score is not None:
                    return float(score)
            
            # If JSON parsing didn't work, return None to trigger retry
            return None
            
        except (json.JSONDecodeError, ValueError, IndexError):
            return None
            
    def get_history(self, node_name: str) -> List[Dict[str, str]]:
        """
        Get conversation history for a specific node
        
        Args:
            node_name: Name of the node
            
        Returns:
            List of conversation history dictionaries
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found")
            
        return self.nodes[node_name].conversation_history
    
    def get_nodes(self) -> List[str]:
        """
        Get list of all current node names
        
        Returns:
            List of node names
        """
        return list(self.nodes.keys())
    
    def get_node_info(self, node_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific node
        
        Args:
            node_name: Name of the node
            
        Returns:
            Dictionary with node information
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found")
            
        node = self.nodes[node_name]
        
        # Include evaluation results if available
        evaluation_info = {}
        if node.evaluation_results:
            evaluation_info = {
                'tests_passed': node.evaluation_results.get('passed', 0),
                'tests_total': node.evaluation_results.get('total', 0),
                'partial_accuracy': node.evaluation_results.get('accuracy', 0.0)
            }
        
        return {
            'name': node.name,
            'problem_id': node.problem_id,
            'prompt_type': node.prompt_type.value,
            'reward': node.reward,
            'response': node.response,
            'code': node.code,  # Uses @property
            'conversation_length': len(node.conversation_history),
            'reward_attempts': len([h for h in node.reward_history if h.get('role') == 'assistant']),
            'children': [child.name for child in node.children],
            'parent': node.parent.name if node.parent else None,
            **evaluation_info  # Include partial test case info
        }
    
    def print_tree_summary(self):
        """Print a summary of the current MCTS tree"""
        print("\n" + "="*60)
        print("🌳 MCTS TREE SUMMARY")
        print("="*60)
        
        if not self.root:
            print("No tree created yet.")
            return
            
        def print_node(node: Node, level: int = 0):
            indent = "  " * level
            print(f"{indent}📍 {node.name} ({node.prompt_type.value})")
            print(f"{indent}   Reward: {node.reward}")
            
            # Show partial success info if available
            if node.evaluation_results:
                passed = node.evaluation_results.get('passed', 0)
                total = node.evaluation_results.get('total', 0)
                accuracy = node.evaluation_results.get('accuracy', 0.0)
                print(f"{indent}   Tests: {passed}/{total} ({accuracy:.1%})")
            
            print(f"{indent}   Children: {len(node.children)}")
            
            for child in node.children:
                print_node(child, level + 1)
        
        print_node(self.root)
        print("="*60) 