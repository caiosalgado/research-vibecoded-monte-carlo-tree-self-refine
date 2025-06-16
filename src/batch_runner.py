#!/usr/bin/env python3
"""
Batch runner for evaluating multiple problems
"""

import json
import time
from datetime import datetime
from .core import get_problem, create_evaluation_prompt, extract_code_delimiters, CodeTester
from aisuite_client import AISuiteClient


def run_batch_evaluation(problem_ids, model="ollama:gemma3:1b", max_problems=None):
    """Run evaluation on multiple problems"""
    
    # Initialize components
    client = AISuiteClient(
        model=model,
        system_prompt="You are an expert Python programmer. Provide clean, efficient code solutions. Follow the exact format requested."
    )
    tester = CodeTester()
    
    # Load all problems if problem_ids is None
    if problem_ids is None:
        with open('data/leetcode_problems.json', 'r') as f:
            data = json.load(f)
        problem_ids = [p['id'] for p in data['problems']]
    
    if max_problems:
        problem_ids = problem_ids[:max_problems]
    
    results = []
    
    print(f"🚀 Starting batch evaluation of {len(problem_ids)} problems")
    print(f"Model: {model}")
    print("=" * 80)
    
    start_time = time.time()
    
    for i, problem_id in enumerate(problem_ids, 1):
        print(f"\n[{i}/{len(problem_ids)}] Processing: {problem_id}")
        
        try:
            # Get problem
            problem = get_problem(problem_id)
            
            # Create prompt and get response
            input_prompt = create_evaluation_prompt(problem)
            response = client.respond(input_prompt, print_response=False)
            code = extract_code_delimiters(response)
            
            if not code:
                results.append({
                    'problem_id': problem_id,
                    'status': 'failed',
                    'error': 'No code extracted from response'
                })
                print(f"❌ Failed: No code extracted")
                continue
            
            # Run evaluations
            partial_eval = tester.run_evaluation(code, problem, 'partial')
            all_eval = tester.run_evaluation(code, problem, 'all')
            
            # Create evaluation result
            evaluation = {
                'problem_id': problem['id'],
                'problem_title': problem['title'],
                'status': 'success',
                'code': code,
                'results': {
                    'full_accuracy': all_eval['accuracy'],
                    'partial_accuracy': partial_eval['accuracy'],
                    'full_details': all_eval,
                    'partial_details': partial_eval,
                }
            }
            
            results.append(evaluation)
            
            # Quick status update
            print(f"✅ Full: {all_eval['accuracy']:.1%} | Partial: {partial_eval['accuracy']:.1%}")
            
        except Exception as e:
            print(f"💥 Exception: {e}")
            results.append({
                'problem_id': problem_id,
                'status': 'exception',
                'error': str(e)
            })
        
        # Small delay to prevent overwhelming the model
        time.sleep(0.5)
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total evaluation time: {total_time:.1f}s")
    
    return results


def generate_report(results):
    """Generate comprehensive evaluation report"""
    if not results:
        print("No results to report!")
        return
    
    successful_results = [r for r in results if r['status'] == 'success']
    
    print("\n" + "="*80)
    print("📊 BATCH EVALUATION REPORT")
    print("="*80)
    
    # Overall Statistics
    total_problems = len(results)
    successful_problems = len(successful_results)
    success_rate = successful_problems / total_problems * 100
    
    print(f"\n🎯 OVERALL STATISTICS:")
    print(f"Total Problems: {total_problems}")
    print(f"Successfully Evaluated: {successful_problems}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if successful_results:
        # Accuracy Statistics
        full_accuracies = [r['results']['full_accuracy'] for r in successful_results]
        partial_accuracies = [r['results']['partial_accuracy'] for r in successful_results]
        
        print(f"\n📈 ACCURACY STATISTICS:")
        print(f"Average Full Accuracy: {sum(full_accuracies)/len(full_accuracies):.1%}")
        print(f"Average Partial Accuracy: {sum(partial_accuracies)/len(partial_accuracies):.1%}")
        
        # Perfect Solutions
        perfect_full = sum(1 for acc in full_accuracies if acc == 1.0)
        perfect_partial = sum(1 for acc in partial_accuracies if acc == 1.0)
        
        print(f"\n🏆 PERFECT SOLUTIONS:")
        print(f"Perfect Full Solutions: {perfect_full}/{successful_problems} ({perfect_full/successful_problems:.1%})")
        print(f"Perfect Partial Solutions: {perfect_partial}/{successful_problems} ({perfect_partial/successful_problems:.1%})")
        
        # Detailed Results
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 80)
        print(f"{'Problem':<25} {'Full':<8} {'Partial':<8} {'Status'}")
        print("-" * 80)
        
        for result in results:
            if result['status'] == 'success':
                r = result['results']
                print(f"{result['problem_title']:<25} {r['full_accuracy']:.1%}     {r['partial_accuracy']:.1%}     ✅")
            else:
                print(f"{result.get('problem_id', 'Unknown'):<25} {'N/A':<8} {'N/A':<8} ❌")
    
    print("\n" + "="*80)


def save_results(results, filename=None):
    """Save results to JSON file"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/batch_evaluation_{timestamp}.json"
    
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'total_problems': len(results),
        'results': results
    }
    
    with open(filename, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"💾 Results saved to: {filename}")
    return filename 