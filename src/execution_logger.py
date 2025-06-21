#!/usr/bin/env python3
"""
Comprehensive Execution Logger for MCTS
Tracks all LLM interactions, evaluations, UCT calculations, and process data
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

from .evaluator import get_problem
from .constants import UCT_EXPLORATION_CONSTANT


class MCTSExecutionLogger:
    """Comprehensive logger for MCTS execution tracking"""
    
    def __init__(self, llm_model: str, system_prompt: str, max_iterations: int, debug_enabled: bool):
        """
        Initialize the execution logger
        
        Args:
            llm_model: Name of the LLM model being used
            system_prompt: System prompt for the LLM
            max_iterations: Maximum iterations for refinement
            debug_enabled: Whether debug mode is enabled
        """
        self.execution_log = {
            "metadata": {
                "start_time": None,
                "end_time": None,
                "duration_seconds": None,
                "model": llm_model,
                "system_prompt": system_prompt,
                "max_iterations": max_iterations,
                "debug_enabled": debug_enabled
            },
            "problem": {},
            "nodes": {},
            "llm_interactions": [],
            "evaluations": [],
            "uct_calculations": [],
            "refinement_cycles": [],
            "backpropagation_steps": [],
            "final_summary": {}
        }
        self.start_time = None
        
    def start_execution(self, problem_id: str):
        """Start logging execution with timing"""
        self.start_time = datetime.now()
        self.execution_log["metadata"]["start_time"] = self.start_time.isoformat()
        
        # Log problem information
        self.execution_log["problem"] = {
            "problem_id": problem_id,
            "problem_data": get_problem(problem_id)
        }
        
    def log_llm_interaction(self, node_name: str, interaction_type: str, prompt: str, response: str, attempt: int = 1):
        """
        Log LLM interaction for comprehensive tracking
        
        Args:
            node_name: Name of the node this interaction belongs to
            interaction_type: Type of interaction ("initial_response", "reward_evaluation", "reflection", "refinement")
            prompt: The prompt sent to the LLM
            response: The response received from the LLM
            attempt: Attempt number (for retry scenarios)
        """
        self.execution_log["llm_interactions"].append({
            "timestamp": datetime.now().isoformat(),
            "node_name": node_name,
            "interaction_type": interaction_type,
            "attempt": attempt,
            "prompt": prompt,
            "response": response,
            "prompt_length": len(prompt),
            "response_length": len(response)
        })
    
    def log_evaluation(self, node_name: str, evaluation_results: Dict[str, Any], reward: float):
        """
        Log code evaluation results
        
        Args:
            node_name: Name of the node being evaluated
            evaluation_results: Results from code evaluation
            reward: Calculated reward value
        """
        self.execution_log["evaluations"].append({
            "timestamp": datetime.now().isoformat(),
            "node_name": node_name,
            "evaluation_results": evaluation_results.copy() if evaluation_results else {},
            "reward": reward,
            "tests_passed": evaluation_results.get('passed', 0) if evaluation_results else 0,
            "tests_total": evaluation_results.get('total', 0) if evaluation_results else 0,
            "accuracy": evaluation_results.get('accuracy', 0.0) if evaluation_results else 0.0
        })
    
    def log_uct_calculation(self, node_name: str, uct_value: float, q_value: float, visits: int, parent_visits: int, is_temp: bool = False):
        """
        Log UCT calculation details
        
        Args:
            node_name: Name of the node
            uct_value: Calculated UCT value
            q_value: Q value used in calculation
            visits: Number of visits to this node
            parent_visits: Number of visits to parent node
            is_temp: Whether this is a temporary calculation (for backpropagation)
        """
        self.execution_log["uct_calculations"].append({
            "timestamp": datetime.now().isoformat(),
            "node_name": node_name,
            "uct_value": uct_value,
            "q_value": q_value,
            "visits": visits,
            "parent_visits": parent_visits,
            "is_temporary": is_temp,
            "exploration_constant": UCT_EXPLORATION_CONSTANT
        })
    
    def log_refinement_cycle(self, iteration: int, selected_node: str, refined_node: str):
        """
        Log refinement cycle information
        
        Args:
            iteration: Refinement iteration number
            selected_node: Name of the node selected for refinement
            refined_node: Name of the newly created refined node
        """
        self.execution_log["refinement_cycles"].append({
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "selected_node": selected_node,
            "refined_node": refined_node
        })
    
    def log_backpropagation_step(self, node_name: str, old_uct: float, new_uct: float, temp_q: Optional[float] = None):
        """
        Log backpropagation step
        
        Args:
            node_name: Name of the node being updated
            old_uct: Previous UCT value
            new_uct: New UCT value after backpropagation
            temp_q: Temporary Q value used (if applicable)
        """
        self.execution_log["backpropagation_steps"].append({
            "timestamp": datetime.now().isoformat(),
            "node_name": node_name,
            "old_uct": old_uct,
            "new_uct": new_uct,
            "temp_q": temp_q
        })
    
    def log_node_selection(self, selected_node: str, uct_value: float, total_nodes: int):
        """
        Log node selection for refinement
        
        Args:
            selected_node: Name of the selected node
            uct_value: UCT value of the selected node
            total_nodes: Total number of nodes available for selection
        """
        self.execution_log["llm_interactions"].append({
            "timestamp": datetime.now().isoformat(),
            "node_name": selected_node,
            "interaction_type": "node_selection",
            "attempt": 1,
            "prompt": f"Selected from {total_nodes} nodes with UCT: {uct_value:.4f}",
            "response": f"Node {selected_node} selected for refinement",
            "prompt_length": 0,
            "response_length": 0
        })
    
    def finalize_and_save(self, nodes: Dict[str, Any], problem_id: str) -> str:
        """
        Finalize logging and save comprehensive log to JSON file
        
        Args:
            nodes: Dictionary of all MCTS nodes
            problem_id: Problem ID for filename
            
        Returns:
            Path to the saved log file
        """
        # End timing
        end_time = datetime.now()
        self.execution_log["metadata"]["end_time"] = end_time.isoformat()
        if self.start_time:
            self.execution_log["metadata"]["duration_seconds"] = (end_time - self.start_time).total_seconds()
        
        # Log all node information
        self.execution_log["nodes"] = {}
        for node_name, node in nodes.items():
            self.execution_log["nodes"][node_name] = {
                "name": node.name,
                "prompt_type": node.prompt_type.value,
                "parent": node.parent.name if node.parent else None,
                "children": [child.name for child in node.children],
                "response": node.response,
                "response_length": len(node.response),
                "code": node.code,
                "code_length": len(node.code),
                "q_list": node.q_list.copy(),
                "final_reward": node.reward,
                "visit_count": node.visit_count,
                "final_uct_value": node.uct_value,
                "conversation_history": node.conversation_history.copy(),
                "reward_history": node.reward_history.copy(),
                "evaluation_results": node.evaluation_results.copy() if node.evaluation_results else None,
                "reflection": getattr(node, 'reflection', ""),
                "parent_node_name": getattr(node, 'parent_node_name', None)
            }
        
        # Generate final summary statistics
        all_nodes = [node for node_name, node in nodes.items() if node_name != "None"]
        if all_nodes:
            best_node = max(all_nodes, key=lambda x: x.reward)
            self.execution_log["final_summary"] = {
                "total_nodes": len(all_nodes),
                "best_node": best_node.name,
                "best_reward": best_node.reward,
                "best_uct": best_node.uct_value,
                "nodes_with_code": sum(1 for node in all_nodes if len(node.code.strip()) > 0),
                "refinement_nodes": sum(1 for node in all_nodes if node.prompt_type.value == "refinement"),
                "total_llm_interactions": len(self.execution_log["llm_interactions"]),
                "total_evaluations": len(self.execution_log["evaluations"]),
                "total_uct_calculations": len(self.execution_log["uct_calculations"]),
                "total_refinement_cycles": len(self.execution_log["refinement_cycles"]),
                "total_backpropagation_steps": len(self.execution_log["backpropagation_steps"]),
                "average_reward": sum(node.reward for node in all_nodes) / len(all_nodes),
                "reward_distribution": {
                    "min": min(node.reward for node in all_nodes),
                    "max": max(node.reward for node in all_nodes),
                    "std": self._calculate_std([node.reward for node in all_nodes])
                }
            }
        else:
            self.execution_log["final_summary"] = {
                "total_nodes": 0,
                "error": "No nodes created"
            }
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mcts_execution_log_{problem_id}_{timestamp}.json"
        
        # Save to results directory
        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", filename)
        
        # Save with pretty formatting
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.execution_log, f, indent=2, ensure_ascii=False)
        
        # Print summary
        self._print_save_summary(filepath)
        
        return filepath
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values"""
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _print_save_summary(self, filepath: str):
        """Print a summary of what was saved"""
        print(f"\n📄 Comprehensive execution log saved to: {filepath}")
        print(f"   📊 Total LLM interactions: {len(self.execution_log['llm_interactions'])}")
        print(f"   🧪 Total evaluations: {len(self.execution_log['evaluations'])}")
        print(f"   📈 Total UCT calculations: {len(self.execution_log['uct_calculations'])}")
        print(f"   🔄 Total refinement cycles: {len(self.execution_log['refinement_cycles'])}")
        print(f"   🔙 Total backpropagation steps: {len(self.execution_log['backpropagation_steps'])}")
        print(f"   🌳 Total nodes: {len(self.execution_log['nodes'])}")
        
        if self.execution_log["final_summary"].get("total_nodes", 0) > 0:
            summary = self.execution_log["final_summary"]
            print(f"   🏆 Best node: {summary['best_node']} (reward: {summary['best_reward']:.2f})")
            print(f"   📊 Average reward: {summary['average_reward']:.2f}")
            print(f"   📈 Reward range: {summary['reward_distribution']['min']:.2f} to {summary['reward_distribution']['max']:.2f}")
    
    def get_log_data(self) -> Dict[str, Any]:
        """Get the current log data (for inspection or testing)"""
        return self.execution_log.copy() 