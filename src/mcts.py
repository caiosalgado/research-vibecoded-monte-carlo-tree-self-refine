#!/usr/bin/env python3
"""
Monte Carlo Tree Search with Self-Refine
Implementation of MCTS for code generation with LLM self-refinement
"""

import json
import re
import math
from enum import Enum
from typing import Dict, List, Optional, Any

from .execution_logger import MCTSExecutionLogger
from .client import AISuiteClient
from .evaluator import CodeTester, extract_code_delimiters, get_problem
from .prompt_templates import create_random_dont_know_prompt, create_weak_answer, call_reward, create_reflection_prompt, create_improvement_prompt
from .constants import REWARD_PARSING_RETRIES, UCT_EXPLORATION_CONSTANT, UCT_EPSILON, MAX_CHILDREN_PER_NODE
from .regex_patterns import regex_parse_score_comprehensive, SCORE_PATTERNS
from .debug_utils import debug_printer, set_debug_mode


class PromptType(Enum):
    """Enum for prompt types to avoid hardcoded strings"""
    ROOT = "root"
    RANDOM_DONT_KNOW = "random_dont_know"
    WEAK_ANSWER = "weak_answer"
    REFINEMENT = "refinement"


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
        self.q_list: List[float] = []  # SSOT for all rewards and visit counting
        self.conversation_history: List[Dict[str, str]] = []
        self.reward_history: List[Dict[str, str]] = []  # Separate reward conversations
        self.evaluation_results: Optional[Dict[str, Any]] = None  # Store evaluation results
        self.uct_value: float = 0.0
        
        # Refinement-specific fields
        self.reflection: str = ""  # Store reflection text for refinement nodes
        self.parent_node_name: Optional[str] = None  # Track which node this was refined from
        
    @property
    def reward(self) -> float:
        """Returns the most recent reward from q_list (SSOT)"""
        return self.q_list[-1] if self.q_list else 0.0
        
    @property
    def visit_count(self) -> int:
        """Returns the number of visits based on q_list length (SSOT)"""
        return len(self.q_list)
        
    @property
    def problem_data(self) -> Dict[str, Any]:
        """Get fresh problem data from single source of truth"""
        return get_problem(self.problem_id)
    
    @property
    def code(self) -> str:
        """Extract code on demand - never stored"""
        return extract_code_delimiters(self.response)
        
    def calculate_uct(self, exploration_constant: float = UCT_EXPLORATION_CONSTANT) -> float:
        """
        Calculate Upper Confidence Bound applied to Trees (UCT) value
        
        UCT_a = Q(a) + c * sqrt(ln(N(Father(a)) + 1) / (N(a) + ε))
        
        Args:
            exploration_constant: Exploration parameter c (default from constants)
            
        Returns:
            UCT value for node selection
        """
        # Q(a) - Average reward using centralized calculation
        exploitation = self.calculate_q_value()
        
        # Get parent visits using centralized method
        parent_visits = self._calculate_parent_visits()
        
        # Calculate exploration term using centralized method
        exploration = self._calculate_exploration_term(exploration_constant, parent_visits, self.visit_count)
        
        uct_value = exploitation + exploration
        self.uct_value = uct_value  # Cache the calculated value
        
        return uct_value
        
    def calculate_uct_with_temp_q(self, temp_q_value: float, exploration_constant: float = UCT_EXPLORATION_CONSTANT) -> float:
        """
        Calculate UCT value using a temporary Q value (for backpropagation)
        
        This method is used during backpropagation to calculate UCT with a temporary
        Q value without modifying the node's actual reward or q_list.
        
        Args:
            temp_q_value: Temporary Q value to use instead of q_list-based calculation
            exploration_constant: Exploration parameter c (default from constants)
            
        Returns:
            UCT value calculated with the temporary Q value
        """
        # Use the temporary Q value instead of calculating from q_list
        exploitation = temp_q_value
        
        # Get parent visits using centralized method
        parent_visits = self._calculate_parent_visits()
        
        # Calculate exploration term using centralized method
        exploration = self._calculate_exploration_term(exploration_constant, parent_visits, self.visit_count)
        
        uct_value = exploitation + exploration
        # Note: We don't update self.uct_value here since this is temporary
        
        return uct_value
        
    def add_child(self, child: 'Node'):
        """Add a child node"""
        child.parent = self
        self.children.append(child)
        
    def _calculate_parent_visits(self) -> int:
        """Centralized parent visits calculation - DRY principle"""
        if self.parent is None:
            return 1  # Root has no parent, use 1 for calculations
        parent_q_list_length = len(self.parent.q_list)
        return parent_q_list_length if parent_q_list_length > 0 else 1
    
    def _calculate_exploration_term(self, exploration_constant: float, parent_visits: int, own_visits: int) -> float:
        """Centralized exploration term calculation - DRY principle"""
        if own_visits == 0:
            return float('inf')  # Unvisited nodes get highest priority
        return exploration_constant * math.sqrt(
            math.log(parent_visits + 1) / (own_visits + UCT_EPSILON)
        )
        
    def calculate_q_value(self) -> float:
        """
        Calculate Q value using the formula: (min(q_list) + avg(q_list)) / 2
        
        This is the centralized location for Q value calculation to avoid duplication.
        
        Returns:
            Q value calculated from q_list, or 0.0 if q_list is empty
        """
        if len(self.q_list) == 0:
            return 0.0  # No rewards yet
        elif len(self.q_list) == 1:
            return self.q_list[0]  # Only one reward, use it directly
        else:
            min_reward = min(self.q_list)
            avg_reward = sum(self.q_list) / len(self.q_list)
            return (min_reward + avg_reward) / 2
        
    def recalculate_and_store_reward(self, mcts_instance):
        """
        Recalculate the reward for this node and append it to q_list
        
        Args:
            mcts_instance: MCTS instance to access reward calculation methods
        """
        # Only recalculate if node has been evaluated before
        if self.evaluation_results is not None:
            new_reward = mcts_instance._get_reward_for_node(self)
            self.q_list.append(new_reward)
            
            if mcts_instance.debug:
                from .debug_utils import debug_printer
                debug_printer.reward_recalculated_and_stored(
                    self.name, new_reward, len(self.q_list)
                )
        
    def to_dict(self):
        """Convert node to dictionary for inspection"""
        return {
            'name': self.name,
            'problem_id': self.problem_id,
            'prompt_type': self.prompt_type.value,
            'reward': self.reward,
            'q_list': self.q_list.copy(),  # Copy to avoid external modification
            'q_list_length': len(self.q_list),  # Keep for backward compatibility
            'visit_count': self.visit_count,  # Now guaranteed to match q_list_length
            'uct_value': self.uct_value,
            'response_length': len(self.response),
            'code_length': len(self.code),  # Uses @property
            'children_count': len(self.children),
            'parent_name': self.parent.name if self.parent else None,
            'reflection_length': len(self.reflection) if self.reflection else 0,
            'parent_node_name': self.parent_node_name
        }


class MCTS:
    """Monte Carlo Tree Search with Self-Refine"""
    
    def __init__(self, llm: AISuiteClient, max_iter: int = 0, debug: bool = False):
        """
        Initialize MCTS
        
        Args:
            llm: AISuiteClient instance for LLM interactions
            max_iter: Maximum iterations (0 for initial step-by-step implementation)
            debug: Enable detailed debug output
        """
        self.llm = llm
        self.max_iter = max_iter
        self.debug = debug
        self.root: Optional[Node] = None
        self.nodes: Dict[str, Node] = {}
        self.problem_id: Optional[str] = None  # Store only ID, not data
        self.tester = CodeTester()
        
        # Set global debug mode
        set_debug_mode(debug)
        
        if self.debug:
            debug_printer.debug_mode_enabled(self.llm.model, self.max_iter)
        
        # Initialize comprehensive execution logger
        self.logger = MCTSExecutionLogger(
            llm_model=llm.model,
            system_prompt=llm.system_prompt,
            max_iterations=max_iter,
            debug_enabled=debug
        )
        
    def fit(self, problem_id: str):
        """
        Main method to run MCTS on a problem
        
        Args:
            problem_id: Problem ID to solve (not the full data)
        """
        # Start execution logging
        self.logger.start_execution(problem_id)
        
        debug_printer.start_mcts()
        
        # Store problem ID only
        self.problem_id = problem_id
        
        # Step 1: Create root node
        self._create_root_node()
        
        # Step 2: Generate two children
        self._generate_initial_children()
        
        # Step 3: Evaluate children
        self._evaluate_children()
        
        # Step 4: Calculate UCT for initial children
        self._calculate_initial_uct()
        
        # Step 5: Iterative refinement cycles
        if self.max_iter > 0:
            debug_printer.start_refinement_cycle_main()
            self._iterative_refinement_cycles()
        
        # Step 6: Generate summary
        self._generate_final_summary()
        
        # Finalize and save execution log
        self.logger.finalize_and_save(self.nodes, problem_id)
        
        debug_printer.mcts_completed()
        
    def _create_root_node(self):
        """Create the root node called 'None'"""
        debug_printer.creating_root_node()
        
        self.root = Node(name="None", problem_id=self.problem_id, prompt_type=PromptType.ROOT)
        self.nodes["None"] = self.root
        
        debug_printer.root_node_created(self.root.name)
        
    def _generate_initial_children(self):
        """Generate two initial children with different prompt strategies"""
        debug_printer.generating_initial_children()
        
        # Child 1: Random "don't know" response
        child1 = self._create_child("root.dk", PromptType.RANDOM_DONT_KNOW)
        
        # Child 2: Weak answer
        child2 = self._create_child("root.wa", PromptType.WEAK_ANSWER)
        
        debug_printer.children_generated(child1.name, len(child1.response), child2.name, len(child2.response))
    
    def _create_child(self, name: str, prompt_type: PromptType) -> Node:
        """Create a child node with given prompt type"""
        # Debug: Section header
        debug_printer.section_creating_child_node(name, prompt_type.value, "None")
        
        debug_printer.generating_response_for_node(name)
        
        # Create node
        child = Node(name=name, problem_id=self.problem_id, prompt_type=prompt_type, parent=self.root)
        
        # Generate response based on prompt type
        if prompt_type == PromptType.RANDOM_DONT_KNOW:
            debug_printer.step_random_dont_know_response()
            
            # For "don't know" responses, the prompt IS the response
            child.response = generate_prompt(child.problem_data, prompt_type)
            child.conversation_history = [
                {"role": "system", "content": self.llm.system_prompt},
                {"role": "assistant", "content": child.response}
            ]
            
            debug_printer.llm_response(child.response, child.name)
                
        else:
            debug_printer.step_generating_prompt(prompt_type.value)
            
            # For other prompt types, generate prompt and get LLM response
            prompt = generate_prompt(child.problem_data, prompt_type)
            debug_printer.prompt_sent(prompt, child.name)
            
            debug_printer.step_sending_prompt_to_llm()
            child.response = self.llm.respond(prompt, print_response=False)
            debug_printer.llm_response(child.response, child.name)
            
            # Log LLM interaction
            self.logger.log_llm_interaction(child.name, "initial_response", prompt, child.response)
            
            child.conversation_history = [
                {"role": "system", "content": self.llm.system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": child.response}
            ]
        
        # Code extraction
        debug_printer.step_extracting_code()
        extracted_code = child.code
        has_code = bool(extracted_code.strip())
        debug_printer.code_extraction(extracted_code, has_code)
        
        if has_code:
            debug_printer.code_extracted_success(name)
        else:
            debug_printer.code_extraction_failed(name)
        
        # Add to tree structure
        self.root.add_child(child)
        self.nodes[name] = child
        
        return child
        
    def _evaluate_children(self):
        """Run reward function for each child"""
        debug_printer.evaluating_children()
        
        for child in self.root.children:
            debug_printer.section_evaluating_node(
                child.name, 
                child.prompt_type.value, 
                child.parent.name if child.parent else None
            )
            
            debug_printer.evaluating_node(child.name)
            
            # Run code evaluation first
            debug_printer.step_running_code_tests()
            evaluation_results = self.tester.run_evaluation(
                child.code,  # Uses @property to extract code on demand
                child.problem_data,  # Uses @property to get fresh problem data
                test_type='partial'
            )
            
            # Store evaluation results in node
            child.evaluation_results = evaluation_results
            debug_printer.test_results(evaluation_results)
            
            # Create reward prompt using the evaluation results
            debug_printer.step_creating_reward_prompt()
            reward_prompt = call_reward(
                child.response, 
                evaluation_results, 
                child.problem_data
            )
            debug_printer.prompt_sent(reward_prompt, child.name, truncate=False)
            
            # Get reward with retry mechanism
            debug_printer.step_parsing_reward()
            reward = self._parse_reward_with_retry(reward_prompt, child)
            
            # Add initial reward to q_list (SSOT for rewards)
            child.q_list.append(reward)
            
            # Log evaluation
            self.logger.log_evaluation(child.name, evaluation_results, reward)
            
            if reward > -100:
                debug_printer.reward_parsed_successfully(reward)
            else:
                debug_printer.reward_parsing_failed_default(reward)
            
            debug_printer.evaluation_results(
                reward, 
                evaluation_results.get('accuracy', 0), 
                evaluation_results.get('passed', 0), 
                evaluation_results.get('total', 0)
            )
    
    def _parse_reward_with_retry(self, reward_prompt: str, child: Node) -> float:
        """Parse reward score from LLM response with retry mechanism"""
        for attempt in range(REWARD_PARSING_RETRIES):
            debug_printer.step_reward_evaluation_attempt(attempt + 1)
            
            # Get reward from LLM  
            reward_response = self.llm.respond(reward_prompt, print_response=False)
            
            # Log reward evaluation interaction
            self.logger.log_llm_interaction(child.name, "reward_evaluation", reward_prompt, reward_response, attempt + 1)
            
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
            parsed_reward = self._parse_reward(reward_response, attempt + 1)
            debug_printer.reward_parsing(reward_response, parsed_reward, child.name, attempt + 1)
            
            if parsed_reward is not None:  # Successfully parsed
                debug_printer.reward_parsed_on_attempt_success(attempt + 1, parsed_reward)
                debug_printer.reward_parsed_success(attempt + 1)
                return parsed_reward
            else:
                debug_printer.parse_attempt_failed(attempt + 1)
                debug_printer.reward_parse_retry(attempt + 1)
        
        # All attempts failed
        debug_printer.all_attempts_failed_default(REWARD_PARSING_RETRIES)
        debug_printer.reward_parse_all_failed(REWARD_PARSING_RETRIES)
        return -101.0
    
    def _parse_reward(self, reward_response: str, attempt: int = 1) -> Optional[float]:
        """Parse reward score from LLM response - returns None if parsing fails"""
        try:
            debug_printer.step_extracting_score_attempt(attempt)
            
            # Use comprehensive regex parsing function
            score = regex_parse_score_comprehensive(reward_response, score_range=(-100, 100))
            
            if score is not None:
                debug_printer.reward_parsed_successfully(score)
                return score
            else:
                debug_printer.no_valid_score_found()
            return None
            
        except (ValueError, IndexError) as e:
            debug_printer.parse_error_failure(str(e))
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
        debug_printer.tree_summary_header()
        
        if not self.root:
            debug_printer.no_tree_created()
            return
            
        def print_node(node: Node, level: int = 0):
            indent = "  " * level
            passed = 0
            total = 0
            accuracy = 0.0
            
            # Get test info if available
            if node.evaluation_results:
                passed = node.evaluation_results.get('passed', 0)
                total = node.evaluation_results.get('total', 0)
                accuracy = node.evaluation_results.get('accuracy', 0.0)
            
            debug_printer.tree_node_info(
                indent, 
                node.name, 
                node.prompt_type.value, 
                node.reward, 
                passed, 
                total, 
                accuracy, 
                len(node.children)
            )
            
            for child in node.children:
                print_node(child, level + 1)
        
        print_node(self.root)
        debug_printer.tree_summary_footer()
        
    def _calculate_initial_uct(self):
        """Calculate UCT values for initial children after evaluation"""
        debug_printer.calculating_initial_uct()
        
        debug_printer.section_uct_calculation_phase()
        
        # Visit counts are now automatically calculated from q_list length in calculate_uct()
        # No need to manually set visit counts
        
        # Calculate UCT for each child
        for child in self.root.children:
            debug_printer.step_calculating_uct_for_node(child.name)
            uct_value = child.calculate_uct()
            
            parent_visits = child.parent.visit_count if child.parent else 0
            debug_printer.uct_calculation(
                child.name, 
                child.reward, 
                child.visit_count, 
                parent_visits, 
                uct_value
            )
            
            # Log UCT calculation
            self.logger.log_uct_calculation(
                child.name, uct_value, child.calculate_q_value(), 
                child.visit_count, parent_visits
            )
            
            debug_printer.uct_result(child.name, uct_value, child.reward, child.visit_count)
            
    def _iterative_refinement_cycles(self):
        """Run iterative MCTS refinement for max_iter iterations"""
        for iteration in range(self.max_iter):
            debug_printer.refinement_iteration_start(iteration + 1, self.max_iter)
            
            # Select best node from ALL nodes
            selected_node = self._select_best_node_global()
            
            # Check if node can be refined
            if not self._can_refine_node(selected_node):
                debug_printer.refinement_iteration_complete(iteration + 1)
                continue
                
            # Create refinement
            refined_node = self._create_refinement(selected_node)
            
            # Log refinement cycle
            self.logger.log_refinement_cycle(iteration + 1, selected_node.name, refined_node.name)
            
            # Evaluate refinement
            self._evaluate_single_node(refined_node)
            
            # Backpropagate UCT values using new algorithm
            self._backpropagate_uct()
            
            debug_printer.refinement_iteration_complete(iteration + 1)

    def _select_best_node_global(self) -> Node:
        """Select node with highest UCT from ALL nodes (excluding root)"""
        eligible_nodes = [node for name, node in self.nodes.items() 
                         if name != "None"]  # All except root
        
        if not eligible_nodes:
            raise ValueError("No eligible nodes for selection")
            
        best_node = max(eligible_nodes, key=lambda n: n.uct_value)
        
        # Recalculate reward for the selected node and append to q_list
        best_node.recalculate_and_store_reward(self)
        
        debug_printer.best_node_selected_global(
            best_node.name, 
            best_node.uct_value, 
            len(eligible_nodes)
        )
        
        # Log node selection
        self.logger.log_node_selection(best_node.name, best_node.uct_value, len(eligible_nodes))
        
        return best_node

    def _create_refined_node_name(self, original_name: str) -> str:
        """Generate name for refined nodes with sequential numbering"""
        # For refinements of refinements, we increment from the original name
        # e.g., root.1.dk.rf_1 -> root.1.dk.rf_2 (not root.1.dk.rf_1.rf_1)
        if ".rf_" in original_name:
            # Extract base name and current refinement number
            parts = original_name.rsplit(".rf_", 1)
            base_name = parts[0]
            current_rf_num = int(parts[1])
            next_rf_num = current_rf_num + 1
        else:
            # First refinement
            base_name = original_name
            next_rf_num = 1
        
        # Find the next available refinement number
        while True:
            candidate_name = f"{base_name}.rf_{next_rf_num}"
            if candidate_name not in self.nodes:
                return candidate_name
            next_rf_num += 1
            
    def _can_refine_node(self, node: Node) -> bool:
        """Check if a node can be refined based on child constraints"""
        num_children = len(node.children)
        
        # Always allow refinement if under max children
        if num_children < MAX_CHILDREN_PER_NODE:
            debug_printer.node_eligible_for_refinement(node.name, num_children, MAX_CHILDREN_PER_NODE)
            return True
            
        # If at max children, check if any child has better Q value
        better_children = sum(1 for child in node.children if child.calculate_q_value() > node.calculate_q_value())
        
        if better_children > 0:
            debug_printer.node_refinement_blocked(
                node.name, num_children, MAX_CHILDREN_PER_NODE, better_children
            )
            return False
        else:
            debug_printer.node_eligible_for_refinement(node.name, num_children, MAX_CHILDREN_PER_NODE)
            return True

    def _create_refinement(self, original_node: Node) -> Node:
        """Create a refinement of any node (combines existing refinement logic)"""
        # Generate reflection
        reflection = self._generate_reflection(original_node)
        
        # Generate improved node
        refined_node = self._generate_refinement(original_node, reflection)
        
        return refined_node

    def _backpropagate_uct(self):
        """
        Backpropagate UCT values using the new algorithm:
        1. Identify leaf nodes (nodes with no children)
        2. Calculate UCT for leaf nodes using their q_list
        3. For each parent/ancestor, update Q using: q_new = (q_current + max(q_children)) / 2
        4. Use temporary Q values to recalculate UCT without modifying q_list
        """
        debug_printer.global_uct_calculation()
        
        # Step 1: Identify all leaf nodes (nodes with no children)
        leaf_nodes = []
        for node_name, node in self.nodes.items():
            if node_name != "None" and len(node.children) == 0:  # Skip root, find leaves
                leaf_nodes.append(node)
        
        debug_printer.backpropagation_started(len(leaf_nodes))
        
        # Step 2: Calculate UCT for leaf nodes using current method (q_list-based)
        for leaf_node in leaf_nodes:
            old_uct = leaf_node.uct_value
            leaf_node.uct_value = leaf_node.calculate_uct()
            debug_printer.node_uct_updated(leaf_node.name, old_uct, leaf_node.uct_value)
            
            # Log backpropagation step for leaf node
            self.logger.log_backpropagation_step(leaf_node.name, old_uct, leaf_node.uct_value)
        
        # Step 3 & 4: Backpropagate to parents and ancestors
        processed_nodes = set()  # Track processed nodes to avoid duplicates
        
        # Start from leaf nodes and work up to root
        nodes_to_process = leaf_nodes.copy()
        
        while nodes_to_process:
            current_node = nodes_to_process.pop(0)
            
            # Process parent if it exists and hasn't been processed
            if current_node.parent and current_node.parent.name not in processed_nodes:
                parent = current_node.parent
                
                # Skip root node
                if parent.name == "None":
                    continue
                
                # Calculate temporary Q value: q_new = (q_current + max(q_children)) / 2
                if len(parent.children) > 0:
                    # Get current Q value from parent's q_list using centralized calculation
                    current_q = parent.calculate_q_value()
                    
                    # Get maximum Q value from children using centralized calculation
                    max_child_q = max(child.calculate_q_value() for child in parent.children)
                    
                    # Calculate new temporary Q value
                    temp_q = (current_q + max_child_q) / 2
                    
                    # Calculate UCT with temporary Q value
                    old_uct = parent.uct_value
                    parent.uct_value = parent.calculate_uct_with_temp_q(temp_q)
                    debug_printer.node_uct_updated(parent.name, old_uct, parent.uct_value)
                    
                    # Log backpropagation step for parent
                    self.logger.log_backpropagation_step(parent.name, old_uct, parent.uct_value, temp_q)
                    
                    debug_printer.backpropagation_temp_q_calculated(parent.name, temp_q, current_q, max_child_q)
                
                # Mark as processed and add to queue for further propagation
                processed_nodes.add(parent.name)
                nodes_to_process.append(parent)

    def _generate_reflection(self, node: Node) -> str:
        """Generate critical reflection for a node"""
        debug_printer.section_reflection_generation(
            node.name,
            node.name,
            node.parent.name if node.parent else None
        )
        
        debug_printer.generating_reflection(node.name)
        
        debug_printer.step_creating_reflection_prompt()
        reflection_prompt = create_reflection_prompt(
            node.problem_data,
            node.response,
            node.evaluation_results
        )
        debug_printer.prompt_sent(reflection_prompt, node.name, truncate=False)
        
        debug_printer.step_getting_reflection_from_llm()
        reflection = self.llm.respond(reflection_prompt, print_response=False)
        debug_printer.llm_response(reflection, node.name, truncate=False)
        
        # Log reflection interaction
        self.logger.log_llm_interaction(node.name, "reflection", reflection_prompt, reflection)
        
        debug_printer.reflection_generated_success(len(reflection))
        debug_printer.reflection_generated(len(reflection))
        
        return reflection
        
    def _generate_refinement(self, original_node: Node, reflection: str) -> Node:
        """Generate a refined node based on reflection"""
        debug_printer.generating_refinement(original_node.name)
        
        # Create refined node name
        refined_name = self._create_refined_node_name(original_node.name)
        
        debug_printer.section_refinement_generation(
            original_node.name,
            refined_name,
            original_node.parent.name if original_node.parent else None
        )
        
        # Create improvement prompt
        debug_printer.step_creating_improvement_prompt()
        improvement_prompt = create_improvement_prompt(
            original_node.problem_data,
            original_node.response,
            original_node.evaluation_results,
            reflection
        )
        debug_printer.prompt_sent(improvement_prompt, refined_name, truncate=False)
        
        # Generate improved response
        debug_printer.step_getting_improved_response()
        improved_response = self.llm.respond(improvement_prompt, print_response=False)
        debug_printer.llm_response(improved_response, refined_name, truncate=False)
        
        # Log refinement interaction
        self.logger.log_llm_interaction(refined_name, "refinement", improvement_prompt, improved_response)
        
        # Create refined node
        refined_node = Node(
            name=refined_name,
            problem_id=self.problem_id,
            prompt_type=PromptType.REFINEMENT,
            parent=original_node  
        )
        
        # Set node properties
        refined_node.response = improved_response
        refined_node.reflection = reflection
        refined_node.parent_node_name = original_node.name
        
        # Set conversation history
        refined_node.conversation_history = [
            {"role": "system", "content": self.llm.system_prompt},
            {"role": "user", "content": improvement_prompt},
            {"role": "assistant", "content": improved_response}
        ]
        
        # Code extraction
        debug_printer.step_extracting_refined_code()
        extracted_code = refined_node.code
        has_code = bool(extracted_code.strip())
        debug_printer.code_extraction(extracted_code, has_code)
        
        if has_code:
            debug_printer.code_extracted_from_refined_success(refined_name)
        else:
            debug_printer.no_code_found_in_refined(refined_name)
        
        # Add to tree structure - refined node should be child of the original selected node
        original_node.add_child(refined_node)
        self.nodes[refined_name] = refined_node
        
        # Visit counts are now automatically calculated from q_list length
        
        debug_printer.refinement_node_created(refined_name)
        return refined_node
        
    def _evaluate_single_node(self, node: Node):
        """Evaluate a single node with reward function"""
        debug_printer.section_single_node_evaluation(
            node.name,
            node.prompt_type.value,
            node.parent.name if node.parent else None
        )
        
        debug_printer.evaluating_single_node(node.name)
        
        # Run code evaluation
        debug_printer.step_running_code_tests()
        evaluation_results = self.tester.run_evaluation(
            node.code,
            node.problem_data,
            test_type='partial'
        )
        
        # Store evaluation results
        node.evaluation_results = evaluation_results
        debug_printer.test_results(evaluation_results)
        
        # Create and process reward
        debug_printer.step_getting_reward_evaluation()
        reward = self._get_reward_for_node(node)
        
        # Add initial reward to q_list (SSOT for rewards)
        node.q_list.append(reward)
        
        # Log evaluation for single node
        self.logger.log_evaluation(node.name, evaluation_results, reward)
        
        if reward > -100:
            debug_printer.reward_parsed_successfully(reward)
        else:
            debug_printer.reward_parsing_failed_default(reward)
        
        debug_printer.single_node_evaluation_results(
            reward, 
            evaluation_results.get('accuracy', 0), 
            evaluation_results.get('passed', 0), 
            evaluation_results.get('total', 0)
        )
        
    def _get_reward_for_node(self, node: Node) -> float:
        """Get reward for a node using the reward function"""
        debug_printer.step_creating_reward_prompt()
        reward_prompt = call_reward(
            node.response,
            node.evaluation_results,
            node.problem_data
        )
        debug_printer.prompt_sent(reward_prompt, node.name, truncate=False)
        
        return self._parse_reward_with_retry(reward_prompt, node)
    
    def _generate_final_summary(self):
        """Generate comprehensive final summary of the MCTS process"""
        debug_printer.section_final_summary()
        
        # Collect all node information
        all_nodes = []
        for node_name in self.get_nodes():
            if node_name != "None":  # Skip root node
                node = self.nodes[node_name]
                node_info = {
                    'name': node.name,
                    'prompt_type': node.prompt_type.value,
                    'reward': node.reward,
                    'uct_value': node.uct_value,
                    'visit_count': node.visit_count,
                    'code_length': len(node.code),
                    'reflection_length': len(node.reflection) if node.reflection else 0,
                    'parent_name': node.parent.name if node.parent else None,
                    'tests_passed': node.evaluation_results.get('passed', 0) if node.evaluation_results else 0,
                    'tests_total': node.evaluation_results.get('total', 0) if node.evaluation_results else 0,
                }
                all_nodes.append(node_info)
        
        # Print summary table
        debug_printer.summary_table(all_nodes)
        
        # Print key insights
        debug_printer.section_process_insights()
        
        # Find best node by reward
        best_node = max(all_nodes, key=lambda x: x['reward']) if all_nodes else None
        if best_node:
            debug_printer.best_performing_node_success(best_node['name'], best_node['reward'])
        
        # Check code extraction success rate
        nodes_with_code = sum(1 for node in all_nodes if node['code_length'] > 0)
        code_success_rate = nodes_with_code / len(all_nodes) if all_nodes else 0
        if code_success_rate >= 0.8:
            debug_printer.code_extraction_stats_success(nodes_with_code, len(all_nodes), code_success_rate)
        else:
            debug_printer.code_extraction_stats_warning(nodes_with_code, len(all_nodes), code_success_rate)
        
        # Check reward parsing success
        valid_rewards = sum(1 for node in all_nodes if node['reward'] > -100)
        reward_success_rate = valid_rewards / len(all_nodes) if all_nodes else 0
        if reward_success_rate >= 0.8:
            debug_printer.reward_parsing_stats_success(valid_rewards, len(all_nodes), reward_success_rate)
        else:
            debug_printer.reward_parsing_stats_warning(valid_rewards, len(all_nodes), reward_success_rate)
        
        # Check if refinement was performed
        refinement_nodes = sum(1 for node in all_nodes if node['prompt_type'] == 'refinement')
        if refinement_nodes > 0:
            debug_printer.refinement_cycle_completed_success(refinement_nodes)
        else:
            debug_printer.no_refinement_cycle_performed()
        
        debug_printer.process_completed(len(all_nodes)) 