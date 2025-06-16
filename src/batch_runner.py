#!/usr/bin/env python3
"""
Batch runner for evaluating multiple problems
"""

import json
import csv
import time
from datetime import datetime
from .evaluator import get_problem, create_evaluation_prompt, extract_code_delimiters, CodeTester
from .client import AISuiteClient


def get_model_name_for_filename(model_string):
    """Extract clean model name for filename"""
    if ":" in model_string:
        parts = model_string.split(":")
        return "-".join(parts[1:]).replace(":", "-")
    return model_string.replace(":", "-")


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


def run_simple_benchmark(model_name="ollama:gemma3:1b"):
    """
    Run comprehensive benchmark with 3 runs per problem
    Automatically generates both CSV and JSON reports
    """
    print("🏁 Starting Challenge Benchmark")
    print("=" * 60)
    
    # Initialize components
    client = AISuiteClient(
        model=model_name,
        system_prompt="You are an expert Python programmer. Provide clean, efficient code solutions. Follow the exact format requested."
    )
    tester = CodeTester()
    
    # Load all problems
    with open('data/leetcode_problems.json', 'r') as f:
        data = json.load(f)
    problems = data['problems']
    
    print(f"Model: {model_name}")
    print(f"Loaded {len(problems)} problems")
    print()
    
    # Generate consistent timestamp for both files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_clean = get_model_name_for_filename(model_name)
    
    # Results storage
    benchmark_results = []
    json_results = []
    
    # Run benchmark for each problem
    for i, problem in enumerate(problems, 1):
        print(f"[{i}/{len(problems)}] Testing: {problem['id']} - {problem['title']}")
        
        # Run 3 evaluations for this problem
        runs = []
        problem_json_runs = []
        
        for run_num in range(1, 4):
            print(f"  Run {run_num}: {problem['id']}")
            
            try:
                # Generate prompt and get LLM response
                prompt = create_evaluation_prompt(problem)
                response = client.respond(prompt, print_response=True)
                code = extract_code_delimiters(response)
                
                if not code:
                    run_result = {
                        'status': 'no_code',
                        'partial_accuracy': 0.0,
                        'full_accuracy': 0.0,
                        'error_count': 1
                    }
                else:
                    # Run evaluations
                    partial_eval = tester.run_evaluation(code, problem, 'partial')
                    full_eval = tester.run_evaluation(code, problem, 'all')
                    
                    run_result = {
                        'status': 'success',
                        'partial_accuracy': partial_eval['accuracy'],
                        'full_accuracy': full_eval['accuracy'],
                        'error_count': len(full_eval['errors'])
                    }
                
                runs.append(run_result)
                
                # Store detailed JSON data
                problem_json_runs.append({
                    'run_number': run_num,
                    'status': run_result['status'],
                    'code': code if code else "",
                    'partial_accuracy': run_result['partial_accuracy'],
                    'full_accuracy': run_result['full_accuracy'],
                    'error_count': run_result['error_count']
                })
                
            except Exception as e:
                print(f"    Error: {str(e)}")
                run_result = {
                    'status': 'error',
                    'partial_accuracy': 0.0,
                    'full_accuracy': 0.0,
                    'error_count': 1
                }
                runs.append(run_result)
                problem_json_runs.append({
                    'run_number': run_num,
                    'status': 'error',
                    'error': str(e),
                    'partial_accuracy': 0.0,
                    'full_accuracy': 0.0
                })
            
            time.sleep(0.5)  # Small delay between runs
        
        # Calculate averages
        avg_partial = sum(r['partial_accuracy'] for r in runs) / 3
        avg_full = sum(r['full_accuracy'] for r in runs) / 3
        total_errors = sum(r['error_count'] for r in runs)
        
        # Store CSV results
        result_row = {
            'problem_id': problem['id'],
            'problem_title': problem['title'],
            'run1_partial': runs[0]['partial_accuracy'],
            'run1_full': runs[0]['full_accuracy'],
            'run1_status': runs[0]['status'],
            'run2_partial': runs[1]['partial_accuracy'],
            'run2_full': runs[1]['full_accuracy'],
            'run2_status': runs[1]['status'],
            'run3_partial': runs[2]['partial_accuracy'],
            'run3_full': runs[2]['full_accuracy'],
            'run3_status': runs[2]['status'],
            'avg_partial_accuracy': avg_partial,
            'avg_full_accuracy': avg_full,
            'total_errors': total_errors,
            'consistency_score': 1.0 - (max(r['full_accuracy'] for r in runs) - min(r['full_accuracy'] for r in runs))
        }
        benchmark_results.append(result_row)
        
        # Store JSON results
        json_results.append({
            'problem_id': problem['id'],
            'problem_title': problem['title'],
            'runs': problem_json_runs,
            'summary': {
                'avg_partial_accuracy': avg_partial,
                'avg_full_accuracy': avg_full,
                'total_errors': total_errors,
                'consistency_score': result_row['consistency_score']
            }
        })
        
        print(f"  Average: Partial {avg_partial:.1%}, Full {avg_full:.1%}")
        print()
    
    # Save CSV results
    csv_filename = f"results/benchmark_results_{model_name_clean}_{timestamp}.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = [
            'problem_id', 'problem_title',
            'run1_partial', 'run1_full', 'run1_status',
            'run2_partial', 'run2_full', 'run2_status', 
            'run3_partial', 'run3_full', 'run3_status',
            'avg_partial_accuracy', 'avg_full_accuracy',
            'total_errors', 'consistency_score'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(benchmark_results)
    
    # Save JSON results
    json_filename = f"results/benchmark_results_{model_name_clean}_{timestamp}.json"
    json_data = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'total_problems': len(problems),
        'benchmark_results': json_results
    }
    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Print summary statistics
    print("=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)
    
    total_problems = len(benchmark_results)
    overall_avg_partial = sum(r['avg_partial_accuracy'] for r in benchmark_results) / total_problems
    overall_avg_full = sum(r['avg_full_accuracy'] for r in benchmark_results) / total_problems
    overall_consistency = sum(r['consistency_score'] for r in benchmark_results) / total_problems
    
    perfect_full = sum(1 for r in benchmark_results if r['avg_full_accuracy'] == 1.0)
    perfect_partial = sum(1 for r in benchmark_results if r['avg_partial_accuracy'] == 1.0)
    
    print(f"Total Problems Tested: {total_problems}")
    print(f"Model Used: {model_name}")
    print(f"Overall Average Partial Accuracy: {overall_avg_partial:.1%}")
    print(f"Overall Average Full Accuracy: {overall_avg_full:.1%}")
    print(f"Overall Consistency Score: {overall_consistency:.1%}")
    print(f"Perfect Full Accuracy: {perfect_full}/{total_problems} ({perfect_full/total_problems:.1%})")
    print(f"Perfect Partial Accuracy: {perfect_partial}/{total_problems} ({perfect_partial/total_problems:.1%})")
    print()
    print(f"📄 Results saved to:")
    print(f"  CSV: {csv_filename}")
    print(f"  JSON: {json_filename}")
    print("🎉 Benchmark Complete!")
    
    return csv_filename, json_filename 