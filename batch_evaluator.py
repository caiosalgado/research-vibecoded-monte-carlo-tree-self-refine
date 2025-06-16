#!/usr/bin/env python3
"""
Batch Evaluator for All Coding Problems
Runs comprehensive evaluation across all problems in the JSON file
"""

import json
import time
from datetime import datetime
from automated_code_evaluator import CodeEvaluator


class BatchEvaluator:
    def __init__(self, model="ollama:gemma3:1b"):
        self.evaluator = CodeEvaluator(model)
        self.results = []
        
    def run_all_problems(self, problems_file='leetcode_problems.json', max_problems=None):
        """Run evaluation on all problems"""
        with open(problems_file, 'r') as f:
            data = json.load(f)
        
        problems = data['problems']
        if max_problems:
            problems = problems[:max_problems]
        
        print(f"🚀 Starting batch evaluation of {len(problems)} problems")
        print(f"Model: {self.evaluator.client.model}")
        print("=" * 80)
        
        start_time = time.time()
        
        for i, problem in enumerate(problems, 1):
            print(f"\n[{i}/{len(problems)}] Processing: {problem['title']}")
            
            try:
                result = self.evaluator.evaluate_problem(problem, print_details=False)
                self.results.append(result)
                
                # Quick status update
                if result['status'] == 'success':
                    results = result['results']
                    print(f"✅ Full: {results['full_accuracy']:.1%} | Partial: {results['partial_accuracy']:.1%} | Hidden: {results['hidden_accuracy']:.1%}")
                else:
                    print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"💥 Exception: {e}")
                self.results.append({
                    'problem_id': problem['id'],
                    'status': 'exception',
                    'error': str(e)
                })
            
            # Small delay to prevent overwhelming the model
            time.sleep(0.5)
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Total evaluation time: {total_time:.1f}s")
        
        return self.results
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        if not self.results:
            print("No results to report!")
            return
        
        successful_results = [r for r in self.results if r['status'] == 'success']
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE EVALUATION REPORT")
        print("="*80)
        
        # Overall Statistics
        total_problems = len(self.results)
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
            hidden_accuracies = [r['results']['hidden_accuracy'] for r in successful_results]
            
            print(f"\n📈 ACCURACY STATISTICS:")
            print(f"Average Full Accuracy: {sum(full_accuracies)/len(full_accuracies):.1%}")
            print(f"Average Partial Accuracy: {sum(partial_accuracies)/len(partial_accuracies):.1%}")
            print(f"Average Hidden Test Accuracy: {sum(hidden_accuracies)/len(hidden_accuracies):.1%}")
            
            # Perfect Solutions
            perfect_full = sum(1 for acc in full_accuracies if acc == 1.0)
            perfect_partial = sum(1 for acc in partial_accuracies if acc == 1.0)
            perfect_hidden = sum(1 for acc in hidden_accuracies if acc == 1.0)
            
            print(f"\n🏆 PERFECT SOLUTIONS:")
            print(f"Perfect Full Solutions: {perfect_full}/{successful_problems} ({perfect_full/successful_problems:.1%})")
            print(f"Perfect Partial Solutions: {perfect_partial}/{successful_problems} ({perfect_partial/successful_problems:.1%})")
            print(f"Perfect Hidden Test: {perfect_hidden}/{successful_problems} ({perfect_hidden/successful_problems:.1%})")
            
            # Detailed Results
            print(f"\n📋 DETAILED RESULTS:")
            print("-" * 80)
            print(f"{'Problem':<25} {'Full':<8} {'Partial':<8} {'Hidden':<8} {'Status'}")
            print("-" * 80)
            
            for result in self.results:
                if result['status'] == 'success':
                    r = result['results']
                    print(f"{result['problem_title']:<25} {r['full_accuracy']:.1%}     {r['partial_accuracy']:.1%}     {r['hidden_accuracy']:.1%}     ✅")
                else:
                    print(f"{result.get('problem_id', 'Unknown'):<25} {'N/A':<8} {'N/A':<8} {'N/A':<8} ❌")
            
            # Error Analysis
            error_results = [r for r in self.results if r['status'] != 'success']
            if error_results:
                print(f"\n❌ ERROR ANALYSIS:")
                for result in error_results[:5]:  # Show first 5 errors
                    print(f"- {result['problem_id']}: {result.get('error', 'Unknown error')}")
                    
        print("\n" + "="*80)
    
    def save_results(self, filename=None):
        """Save results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'model': self.evaluator.client.model,
            'total_problems': len(self.results),
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")
        return filename


def main():
    """Run batch evaluation"""
    evaluator = BatchEvaluator()
    
    # Run evaluation (limit to 3 problems for testing)
    print("🧪 Running evaluation on first 3 problems for testing...")
    results = evaluator.run_all_problems(max_problems=3)
    
    # Generate report
    evaluator.generate_report()
    
    # Save results
    evaluator.save_results()
    
    print("\n🎉 Batch evaluation complete!")


if __name__ == "__main__":
    main() 