#!/usr/bin/env python3
"""
Debug utilities for MCTS debugging with consistent formatting
"""

from typing import Dict, Any, Optional, List
import json
from datetime import datetime


class DebugPrinter:
    """Standardized debug printing with consistent formatting"""
    
    # Node type emojis
    NODE_EMOJIS = {
        'root': '🌳',
        'random_dont_know': '🤷',
        'weak_answer': '🌿', 
        'refinement': '🔄'
    }
    
    # Step emojis
    STEP_EMOJIS = {
        'prompt': '📤',
        'response': '📥',
        'extraction': '🔍',
        'evaluation': '🧪',
        'reward': '🎯',
        'uct': '📈',
        'reflection': '💭',
        'improvement': '🔧',
        'success': '✅',
        'failure': '❌',
        'warning': '⚠️'
    }
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.step_counter = 0
        
    def section_header(self, title: str, node_name: str = None, node_type: str = None, parent_name: str = None):
        """Print a standardized section header"""
        if not self.enabled:
            return
            
        self.step_counter += 1
        
        print("\n" + "="*80)
        
        if node_name:
            emoji = self.NODE_EMOJIS.get(node_type, '🔵')
            print(f"{emoji} NODE: {node_name}")
            if parent_name:
                print(f"   👆 Parent: {parent_name}")
            if node_type:
                print(f"   🏷️  Type: {node_type}")
        
        print(f"📋 STEP {self.step_counter}: {title}")
        print("="*80)
        
    def step_start(self, step_name: str, details: str = None):
        """Print step start with emoji"""
        if not self.enabled:
            return
            
        emoji = self.STEP_EMOJIS.get(step_name.lower(), '🔵')
        print(f"\n{emoji} {step_name.upper()}")
        if details:
            print(f"   {details}")
        print("-" * 60)
        
    def prompt_sent(self, prompt: str, truncate: bool = False):
        """Print sent prompt with formatting"""
        if not self.enabled:
            return
            
        print("📤 PROMPT SENT TO LLM:")
        print("┌" + "─" * 58 + "┐")
        
        if truncate and len(prompt) > 500:
            lines = prompt.split('\n')
            if len(lines) > 20:
                displayed_lines = lines[:10] + ['...'] + lines[-10:]
                prompt_display = '\n'.join(displayed_lines)
            else:
                prompt_display = prompt[:500] + "..." if len(prompt) > 500 else prompt
        else:
            prompt_display = prompt
            
        for line in prompt_display.split('\n'):
            print(f"│ {line:<56} │")
        print("└" + "─" * 58 + "┘")
        
    def llm_response(self, response: str, truncate: bool = False):
        """Print LLM response with formatting"""
        if not self.enabled:
            return
            
        print("📥 LLM RESPONSE:")
        print("┌" + "─" * 58 + "┐")
        
        if truncate and len(response) > 500:
            lines = response.split('\n')
            if len(lines) > 20:
                displayed_lines = lines[:10] + ['...'] + lines[-10:]
                response_display = '\n'.join(displayed_lines)
            else:
                response_display = response[:500] + "..." if len(response) > 500 else response
        else:
            response_display = response
            
        for line in response_display.split('\n'):
            print(f"│ {line:<56} │")
        print("└" + "─" * 58 + "┘")
        
    def code_extraction(self, code: str, success: bool):
        """Print code extraction result"""
        if not self.enabled:
            return
            
        if success and code.strip():
            print("🔍 CODE EXTRACTION: ✅ SUCCESS")
            print("```python")
            print(code)
            print("```")
        else:
            print("🔍 CODE EXTRACTION: ❌ FAILED")
            print("   No code found or empty extraction")
            
    def test_results(self, results: Dict[str, Any]):
        """Print test evaluation results"""
        if not self.enabled:
            return
            
        print("🧪 TEST EVALUATION RESULTS:")
        if results:
            passed = results.get('passed', 0)
            total = results.get('total', 0)
            accuracy = results.get('accuracy', 0)
            
            print(f"   📊 Tests Passed: {passed}/{total}")
            print(f"   📊 Accuracy: {accuracy:.1%}")
            
            errors = results.get('errors', [])
            if errors:
                print("   ❌ Errors Found:")
                for i, error in enumerate(errors[:3], 1):
                    error_type = error.get('error_type', 'unknown')
                    expected = error.get('expected', 'N/A')
                    actual = error.get('actual', 'N/A')
                    print(f"      {i}. {error_type}: Expected {expected}, Got {actual}")
        else:
            print("   ❌ No evaluation results available")
            
    def reward_parsing(self, raw_response: str, parsed_value: Optional[float], attempt: int = 1):
        """Print reward parsing attempt"""
        if not self.enabled:
            return
            
        print(f"🎯 REWARD PARSING (Attempt {attempt}):")
        print("Raw LLM Response:")
        print("┌" + "─" * 38 + "┐")
        for line in raw_response.split('\n')[:10]:  # Show first 10 lines
            print(f"│ {line:<36} │")
        if len(raw_response.split('\n')) > 10:
            print("│ ...                                │")
        print("└" + "─" * 38 + "┘")
        
        if parsed_value is not None:
            print(f"✅ Parsed Reward: {parsed_value}")
        else:
            print("❌ Parsing Failed")
            
    def uct_calculation(self, node_name: str, reward: float, visits: int, parent_visits: int, uct_value: float):
        """Print UCT calculation details"""
        if not self.enabled:
            return
            
        print(f"📈 UCT CALCULATION for {node_name}:")
        print(f"   Reward (Q): {reward:.4f}")
        print(f"   Own visits (N): {visits}")
        print(f"   Parent visits: {parent_visits}")
        print(f"   UCT Value: {uct_value:.4f}")
        
    def success(self, message: str):
        """Print success message"""
        if not self.enabled:
            return
        print(f"✅ SUCCESS: {message}")
        
    def failure(self, message: str):
        """Print failure message"""
        if not self.enabled:
            return
        print(f"❌ FAILURE: {message}")
        
    def warning(self, message: str):
        """Print warning message"""
        if not self.enabled:
            return
        print(f"⚠️  WARNING: {message}")
        
    def summary_table(self, nodes: List[Dict[str, Any]]):
        """Print a summary table of all nodes"""
        if not self.enabled:
            return
            
        print("\n" + "="*100)
        print("📊 FINAL SUMMARY TABLE")
        print("="*100)
        
        # Table header
        print(f"{'Node Name':<20} {'Type':<12} {'Reward':<8} {'UCT':<8} {'Code?':<6} {'Reflect?':<9} {'Tests':<10} {'Parent':<15}")
        print("-" * 100)
        
        # Table rows
        for node in nodes:
            name = node.get('name', 'N/A')[:19]
            node_type = node.get('prompt_type', 'N/A')[:11]
            reward = f"{node.get('reward', 0):.2f}"
            uct = f"{node.get('uct_value', 0):.2f}"
            has_code = "✅" if node.get('code_length', 0) > 0 else "❌"
            has_reflection = "✅" if node.get('reflection_length', 0) > 0 else "❌"
            tests = f"{node.get('tests_passed', 0)}/{node.get('tests_total', 0)}"
            parent = node.get('parent_name', 'None')[:14]
            
            print(f"{name:<20} {node_type:<12} {reward:<8} {uct:<8} {has_code:<6} {has_reflection:<9} {tests:<10} {parent:<15}")
        
        print("="*100)
        
    def navigation(self, from_node: str, to_node: str, action: str):
        """Print navigation between nodes"""
        if not self.enabled:
            return
        print(f"🧭 NAVIGATION: {action} from {from_node} to {to_node}")
    
    def progress(self, message: str, always_show: bool = True):
        """Print progress message - shows even when debug disabled if always_show=True"""
        if always_show or self.enabled:
            print(message)
    
    def result(self, message: str, always_show: bool = False):
        """Print result message - only shows in debug mode unless always_show=True"""
        if always_show or self.enabled:
            print(message)
    
    # === MCTS WORKFLOW METHODS ===
    
    def debug_mode_enabled(self, model: str, max_iter: int):
        """Show debug mode activation"""
        if not self.enabled:
            return
        self.progress("🐛 DEBUG MODE ENABLED - Comprehensive logging active")
        self.progress(f"   🤖 Model: {model}")
        self.progress(f"   🔄 Max iterations: {max_iter}")
        self.progress("")
    
    def start_mcts(self):
        """Starting MCTS fit process"""
        self.progress("🌳 Starting MCTS fit process...")
    
    def start_refinement_cycle_main(self):
        """Starting refinement cycle (main)"""
        self.progress("🔄 Starting refinement cycle...")
    
    def mcts_completed(self):
        """MCTS fit process completed"""
        self.progress("✅ MCTS fit process completed!")
    
    def creating_root_node(self):
        """Creating root node"""
        self.progress("📍 Creating root node...")
    
    def root_node_created(self, node_name: str):
        """Root node created successfully"""
        self.result(f"   ✓ Root node '{node_name}' created")
    
    def generating_initial_children(self):
        """Generating initial children"""
        self.progress("👶 Generating initial children...")
    
    def children_generated(self, child1_name: str, child1_response_len: int, child2_name: str, child2_response_len: int):
        """Initial children generated"""
        self.result(f"   ✓ Child 1: {child1_name} (Response: {child1_response_len} chars)")
        self.result(f"   ✓ Child 2: {child2_name} (Response: {child2_response_len} chars)")
    
    def generating_response_for_node(self, node_name: str):
        """Generating response for specific node"""
        self.progress(f"   🧠 Generating response for {node_name}...", always_show=False)
    
    def evaluating_children(self):
        """Evaluating children with reward function"""
        self.progress("🏆 Evaluating children with reward function...")
    
    def evaluating_node(self, node_name: str):
        """Evaluating specific node"""
        self.progress(f"   📊 Evaluating {node_name}...", always_show=False)
    
    def evaluation_results(self, reward: float, accuracy: float, passed: int, total: int):
        """Show evaluation results for a node"""
        self.result(f"      ✓ Reward: {reward}")
        self.result(f"      ✓ Accuracy: {accuracy:.1%}")
        self.result(f"      ✓ Tests passed: {passed}/{total}")
    
    def reward_parsed_success(self, attempt: int):
        """Reward parsing succeeded"""
        self.result(f"     ✓ Reward parsed successfully on attempt {attempt}")
    
    def reward_parse_retry(self, attempt: int):
        """Reward parsing failed, retrying"""
        self.result(f"     ⚠️  Parse attempt {attempt} failed, retrying...")
    
    def reward_parse_all_failed(self, max_retries: int):
        """All reward parsing attempts failed"""
        self.result(f"     ❌ All {max_retries} attempts failed, setting reward to -101")
    
    def calculating_initial_uct(self):
        """Calculating UCT values for initial children"""
        self.progress("📊 Calculating UCT values for initial children...")
    
    def uct_result(self, node_name: str, uct_value: float, reward: float, visits: int):
        """Show UCT calculation result"""
        self.result(f"   📈 {node_name}: UCT = {uct_value:.4f} (reward: {reward:.4f}, visits: {visits})")
    
    def best_node_selected(self, node_name: str, uct_value: float):
        """Best node selected by UCT"""
        self.result(f"🎯 Selected best node: {node_name} (UCT: {uct_value:.4f})")
    
    def starting_refinement_cycle(self):
        """Starting refinement cycle (detailed)"""
        self.progress("🔍 Starting refinement cycle...", always_show=False)
    
    def refined_node_created(self, node_name: str, uct_value: float, reward: float):
        """Refined node created with metrics"""
        self.result(f"🔥 Refined node: {node_name}")
        self.result(f"   📈 UCT = {uct_value:.4f} (reward: {reward:.4f})")
    
    def generating_reflection(self, node_name: str):
        """Generating reflection for node"""
        self.progress(f"💭 Generating reflection for {node_name}...", always_show=False)
    
    def reflection_generated(self, length: int):
        """Reflection generated successfully"""
        self.result(f"   ✓ Reflection generated ({length} chars)")
    
    def generating_refinement(self, node_name: str):
        """Generating refinement for node"""
        self.progress(f"🔧 Generating refinement for {node_name}...", always_show=False)
    
    def refinement_node_created(self, node_name: str):
        """Refined node created"""
        self.result(f"   ✓ Refined node created: {node_name}")
    
    def evaluating_single_node(self, node_name: str):
        """Evaluating single node"""
        self.progress(f"🏆 Evaluating {node_name}...", always_show=False)
    
    def single_node_evaluation_results(self, reward: float, accuracy: float, passed: int, total: int):
        """Single node evaluation results"""
        self.result(f"   📊 Reward: {reward:.4f}")
        self.result(f"   📊 Accuracy: {accuracy:.1%}")
        self.result(f"   📊 Tests passed: {passed}/{total}")
    
    def process_completed(self, total_nodes: int):
        """Process completed with total nodes"""
        self.progress(f"\n📊 Process completed with {total_nodes} total nodes")
    
    # === TREE SUMMARY METHODS ===
    
    def tree_summary_header(self):
        """Tree summary header"""
        self.progress("\n" + "="*60)
        self.progress("🌳 MCTS TREE SUMMARY")
        self.progress("="*60)
    
    def tree_summary_footer(self):
        """Tree summary footer"""
        self.progress("="*60)
    
    def no_tree_created(self):
        """No tree created yet"""
        self.progress("No tree created yet.")
    
    def tree_node_info(self, indent: str, node_name: str, node_type: str, reward: float, passed: int, total: int, accuracy: float, children_count: int):
        """Show tree node information"""
        self.progress(f"{indent}📍 {node_name} ({node_type})")
        self.progress(f"{indent}   Reward: {reward}")
        if total > 0:
            self.progress(f"{indent}   Tests: {passed}/{total} ({accuracy:.1%})")
        self.progress(f"{indent}   Children: {children_count}")
    
    # === STEP-SPECIFIC METHODS ===
    
    def step_random_dont_know_response(self):
        """Step: Random 'don't know' response"""
        self.step_start("prompt_generation", "Random 'don't know' response")
    
    def step_generating_prompt(self, prompt_type_value: str):
        """Step: Generating specific prompt type"""
        self.step_start("prompt_generation", f"Generating {prompt_type_value} prompt")
    
    def step_sending_prompt_to_llm(self):
        """Step: Sending prompt to LLM"""
        self.step_start("llm_call", "Sending prompt to LLM")
    
    def step_extracting_code(self):
        """Step: Extracting code from response"""
        self.step_start("extraction", "Extracting code from response")
    
    def step_running_code_tests(self):
        """Step: Running code tests"""
        self.step_start("evaluation", "Running code tests")
    
    def step_creating_reward_prompt(self):
        """Step: Creating reward evaluation prompt"""
        self.step_start("reward", "Creating reward evaluation prompt")
    
    def step_parsing_reward(self):
        """Step: Parsing reward from LLM"""
        self.step_start("reward_parsing", "Parsing reward from LLM")
    
    def step_reward_evaluation_attempt(self, attempt: int):
        """Step: Reward evaluation attempt"""
        self.step_start("llm_call", f"Reward evaluation attempt {attempt}")
    
    def step_extracting_score_attempt(self, attempt: int):
        """Step: Attempting to extract score"""
        self.step_start("parsing", f"Attempting to extract score (attempt {attempt})")
    
    def step_json_parsing_found(self, json_preview: str):
        """Step: Found JSON-like structure"""
        self.step_start("json_parsing", f"Found JSON-like structure: {json_preview}")
    
    def step_calculating_uct_for_node(self, node_name: str):
        """Step: Calculating UCT for specific node"""
        self.step_start("uct", f"Calculating UCT for {node_name}")
    
    def step_creating_reflection_prompt(self):
        """Step: Creating critical reflection prompt"""
        self.step_start("reflection", "Creating critical reflection prompt")
    
    def step_getting_reflection_from_llm(self):
        """Step: Getting reflection from LLM"""
        self.step_start("llm_call", "Getting reflection from LLM")
    
    def step_creating_improvement_prompt(self):
        """Step: Creating improvement prompt with reflection"""
        self.step_start("improvement", "Creating improvement prompt with reflection")
    
    def step_getting_improved_response(self):
        """Step: Getting improved response from LLM"""
        self.step_start("llm_call", "Getting improved response from LLM")
    
    def step_extracting_refined_code(self):
        """Step: Extracting code from refined response"""
        self.step_start("extraction", "Extracting code from refined response")
    
    def step_getting_reward_evaluation(self):
        """Step: Getting reward evaluation"""
        self.step_start("reward_parsing", "Getting reward evaluation")
    
    # === SECTION HEADER METHODS ===
    
    def section_uct_calculation_phase(self):
        """Section: UCT Calculation Phase"""
        self.section_header("UCT Calculation Phase", "Calculating UCT for all children")
    
    def section_final_summary(self):
        """Section: Final Summary"""
        self.section_header("FINAL SUMMARY", "Complete MCTS Process Summary")
    
    def section_process_insights(self):
        """Section: Process insights"""
        self.step_start("insights", "Key Process Insights")
    
    # === SPECIFIC FAILURE METHODS ===
    
    def no_valid_score_found(self):
        """No valid score found in response"""
        self.failure("No valid score found in response")
    
    def no_refinement_cycle_performed(self):
        """No refinement cycle performed"""
        self.warning("No refinement cycle performed")
    
    # === SUCCESS/FAILURE/WARNING METHODS ===
    
    def code_extracted_success(self, node_name: str):
        """Code extraction successful"""
        self.success(f"Code extracted from {node_name}")
    
    def code_extraction_failed(self, node_name: str):
        """Code extraction failed"""
        self.failure(f"No code found in {node_name}")
    
    def reward_parsed_successfully(self, reward: float):
        """Reward parsed successfully"""
        self.success(f"Reward parsed successfully: {reward}")
    
    def reward_parsing_failed_default(self, reward: float):
        """Reward parsing failed, using default"""
        self.failure(f"Reward parsing failed, using default: {reward}")
    
    def reward_parsed_on_attempt_success(self, attempt: int, parsed_reward: float):
        """Reward parsed successfully on specific attempt"""
        self.success(f"Reward parsed on attempt {attempt}: {parsed_reward}")
    
    def parse_attempt_failed(self, attempt: int):
        """Parse attempt failed"""
        self.failure(f"Parse attempt {attempt} failed")
    
    def all_attempts_failed_default(self, max_retries: int):
        """All attempts failed, using default"""
        self.failure(f"All {max_retries} attempts failed, using default -101")
    
    def score_extracted_using_pattern(self, pattern_num: int, score: float):
        """Score extracted using specific pattern"""
        self.success(f"Score extracted using pattern {pattern_num}: {score}")
    
    def score_extracted_from_json_pattern(self, score: float):
        """Score extracted from JSON pattern"""
        self.success(f"Score extracted from JSON pattern: {score}")
    
    def score_extracted_from_json_parsing(self, score: float):
        """Score extracted from JSON parsing"""
        self.success(f"Score extracted from JSON parsing: {score}")
    
    def json_parse_error_warning(self, error: str):
        """JSON parse error warning"""
        self.warning(f"JSON parse error: {error}")
    
    def using_fallback_number_extraction(self, score: float):
        """Using fallback number extraction"""
        self.warning(f"Using fallback number extraction: {score}")
    
    def parse_error_failure(self, error: str):
        """Parse error failure"""
        self.failure(f"Parse error: {error}")
    
    def reflection_generated_success(self, length: int):
        """Reflection generated successfully"""
        self.success(f"Reflection generated ({length} chars)")
    
    def code_extracted_from_refined_success(self, refined_name: str):
        """Code extracted from refined node successfully"""
        self.success(f"Code extracted from {refined_name}")
    
    def no_code_found_in_refined(self, refined_name: str):
        """No code found in refined node"""
        self.failure(f"No code found in {refined_name}")
    
    def best_performing_node_success(self, node_name: str, reward: float):
        """Best performing node identified"""
        self.success(f"Best performing node: {node_name} (reward: {reward:.2f})")
    
    def code_extraction_stats_success(self, nodes_with_code: int, total_nodes: int, success_rate: float):
        """Code extraction statistics - success"""
        self.success(f"Code extraction: {nodes_with_code}/{total_nodes} nodes ({success_rate:.1%})")
    
    def code_extraction_stats_warning(self, nodes_with_code: int, total_nodes: int, success_rate: float):
        """Code extraction statistics - warning"""
        self.warning(f"Code extraction: {nodes_with_code}/{total_nodes} nodes ({success_rate:.1%})")
    
    def reward_parsing_stats_success(self, valid_rewards: int, total_nodes: int, success_rate: float):
        """Reward parsing statistics - success"""
        self.success(f"Reward parsing: {valid_rewards}/{total_nodes} nodes ({success_rate:.1%})")
    
    def reward_parsing_stats_warning(self, valid_rewards: int, total_nodes: int, success_rate: float):
        """Reward parsing statistics - warning"""
        self.warning(f"Reward parsing: {valid_rewards}/{total_nodes} nodes ({success_rate:.1%})")
    
    def refinement_cycle_completed_success(self, refinement_nodes: int):
        """Refinement cycle completed successfully"""
        self.success(f"Refinement cycle completed: {refinement_nodes} refined nodes created")
    
    # === SECTION HEADER METHODS FOR F-STRINGS ===
    
    def section_creating_child_node(self, name: str, node_type: str, parent_name: str):
        """Section header for creating child node"""
        self.section_header(
            f"Creating Child Node: {name}",
            node_name=name,
            node_type=node_type,
            parent_name=parent_name
        )
    
    def section_evaluating_node(self, node_name: str, node_type: str, parent_name: str):
        """Section header for evaluating node"""
        self.section_header(
            f"Evaluating Node: {node_name}",
            node_name=node_name,
            node_type=node_type,
            parent_name=parent_name
        )
    
    def section_reflection_generation(self, node_name: str, refined_name: str, parent_name: str):
        """Section header for reflection generation"""
        self.section_header(
            f"Reflection Generation for {node_name}",
            node_name=refined_name,
            node_type="refinement", 
            parent_name=parent_name
        )
    
    def section_refinement_generation(self, original_node_name: str, refined_name: str, parent_name: str):
        """Section header for refinement generation"""
        self.section_header(
            f"Refinement Generation for {original_node_name}",
            node_name=refined_name,
            node_type="refinement",
            parent_name=parent_name
        )
    
    def section_single_node_evaluation(self, node_name: str, node_type: str, parent_name: str):
        """Section header for single node evaluation"""
        self.section_header(
            f"Single Node Evaluation: {node_name}",
            node_name=node_name,
            node_type=node_type,
            parent_name=parent_name
        )
    
    def create_refined_node_name(self, original_name: str) -> str:
        """Create refined node name"""
        return f"{original_name}.refined"


# Global debug printer instance
debug_printer = DebugPrinter(enabled=False)  # Will be enabled when debug mode is on


def set_debug_mode(enabled: bool):
    """Enable or disable debug mode globally"""
    global debug_printer
    debug_printer.enabled = enabled 