#!/usr/bin/env python3
"""
Evaluation System Demo
Shows different ways to use the automated code evaluation system
"""

import json
from automated_code_evaluator import CodeEvaluator
from batch_evaluator import BatchEvaluator


def demo_single_problem():
    """Demo: Evaluate a single problem with detailed output"""
    print("🔍 DEMO 1: Single Problem Evaluation")
    print("=" * 60)
    
    # Load problems
    with open('leetcode_problems.json', 'r') as f:
        data = json.load(f)
    
    # Pick the Palindrome problem
    palindrome_problem = next(p for p in data['problems'] if p['id'] == 'isPalindrome')
    
    evaluator = CodeEvaluator()
    result = evaluator.evaluate_problem(palindrome_problem, print_details=True)
    
    return result


def demo_batch_evaluation():
    """Demo: Batch evaluation of multiple problems"""
    print("\n\n🔄 DEMO 2: Batch Evaluation")
    print("=" * 60)
    
    batch_evaluator = BatchEvaluator()
    
    # Run on first 2 problems
    results = batch_evaluator.run_all_problems(max_problems=2)
    batch_evaluator.generate_report()
    
    return results


def demo_custom_analysis():
    """Demo: Custom analysis of results"""
    print("\n\n📊 DEMO 3: Custom Analysis")
    print("=" * 60)
    
    # Load a saved results file (from previous run)
    try:
        # Try to find a results file
        import glob
        results_files = glob.glob("evaluation_results_*.json")
        if results_files:
            latest_file = max(results_files)
            print(f"Loading results from: {latest_file}")
            
            with open(latest_file, 'r') as f:
                saved_data = json.load(f)
            
            results = saved_data['results']
            
            # Custom analysis
            print(f"\n📈 CUSTOM ANALYSIS:")
            
            # Find problems that had perfect generalization (hidden test)
            perfect_generalization = [
                r for r in results 
                if r['status'] == 'success' and r['results']['hidden_accuracy'] == 1.0
            ]
            
            print(f"Problems with perfect generalization: {len(perfect_generalization)}")
            for result in perfect_generalization:
                print(f"  - {result['problem_title']}")
            
            # Find problems where visible tests passed but hidden failed
            generalization_gap = [
                r for r in results 
                if r['status'] == 'success' and 
                r['results']['partial_accuracy'] > r['results']['hidden_accuracy']
            ]
            
            print(f"\nProblems with generalization gap: {len(generalization_gap)}")
            for result in generalization_gap:
                print(f"  - {result['problem_title']}: Visible {result['results']['partial_accuracy']:.1%} vs Hidden {result['results']['hidden_accuracy']:.1%}")
                
        else:
            print("No previous results found. Run batch evaluation first.")
            
    except Exception as e:
        print(f"Error loading results: {e}")


def demo_different_models():
    """Demo: Compare different models (if available)"""
    print("\n\n🤖 DEMO 4: Model Comparison")
    print("=" * 60)
    
    # Available models (modify based on what you have)
    available_models = [
        "ollama:gemma3:1b", 
        # "ollama:qwen3:1.7b",  # Uncomment if you want to test
    ]
    
    # Load one simple problem
    with open('leetcode_problems.json', 'r') as f:
        data = json.load(f)
    
    problem = data['problems'][0]  # First problem
    
    print(f"Testing problem: {problem['title']}")
    print("-" * 40)
    
    for model in available_models:
        try:
            print(f"\n🔥 Testing with {model}:")
            evaluator = CodeEvaluator(model)
            result = evaluator.evaluate_problem(problem, print_details=False)
            
            if result['status'] == 'success':
                r = result['results']
                print(f"  Full: {r['full_accuracy']:.1%} | Partial: {r['partial_accuracy']:.1%} | Hidden: {r['hidden_accuracy']:.1%}")
            else:
                print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"  💥 Error: {e}")


def main():
    """Run all demos"""
    print("🎯 AUTOMATED CODE EVALUATION SYSTEM DEMO")
    print("=" * 80)
    
    # Demo 1: Single problem
    demo_single_problem()
    
    # Demo 2: Batch evaluation
    demo_batch_evaluation()
    
    # Demo 3: Custom analysis
    demo_custom_analysis()
    
    # Demo 4: Model comparison
    demo_different_models()
    
    print("\n" + "=" * 80)
    print("🎉 ALL DEMOS COMPLETE!")
    print("\n💡 USAGE TIPS:")
    print("1. Use CodeEvaluator for single problems with detailed analysis")
    print("2. Use BatchEvaluator for comprehensive testing across all problems")
    print("3. Modify models in the constructors to test different LLMs")
    print("4. Check saved JSON files for detailed results and error analysis")
    print("5. The system handles both simple and complex problems (including LinkedList)")


if __name__ == "__main__":
    main() 