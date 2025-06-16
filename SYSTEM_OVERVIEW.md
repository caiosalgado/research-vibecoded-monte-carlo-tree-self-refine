# 🎯 Automated Code Evaluation System

## Overview
A comprehensive system that uses AISuite to generate code solutions and automatically evaluates them against test cases with both full and partial (generalization) testing.

## 🏗️ System Architecture

### Core Components

#### 1. **AISuiteClient** (`aisuite_client.py`)
- Simple wrapper around AISuite library
- Configurable system prompts and models
- Clean interface for getting AI responses

#### 2. **AutomatedTester** (`automated_code_evaluator.py`)
- Handles code execution and testing
- Supports complex data types (LinkedList, etc.)
- Provides detailed error reporting and accuracy metrics

#### 3. **CodeEvaluator** (`automated_code_evaluator.py`)
- Main evaluation engine
- Creates structured prompts with hidden test cases
- Extracts code using regex patterns
- Runs comprehensive evaluations

#### 4. **BatchEvaluator** (`batch_evaluator.py`)
- Processes multiple problems automatically
- Generates comprehensive reports
- Saves results to JSON for later analysis

## 🎨 Hybrid Approach Implementation

### Structured Prompts (Approach 1)
```python
def create_evaluation_prompt(self, problem_data):
    visible_tests = problem_data['tests'][:-1]  # Hide last test
    
    prompt = f"""Solve this coding problem:
    
    **Problem:** {problem_data['title']}
    **Description:** {problem_data['description']}
    **Visible Test Cases:** {visible_tests}
    **Note:** There is 1 additional hidden test case for evaluation.
    
    **CRITICAL FORMATTING:** Provide your solution in this EXACT format:
    ```python
    {problem_data['function_signature']}
        # Your implementation here
        pass
    ```
    
    Respond with ONLY the function code block. No explanations."""
```

### Comprehensive Testing Framework (Approach 4)
- **Code Extraction**: Reliable regex-based extraction with fallbacks
- **Dynamic Execution**: Safe code execution with proper namespace setup
- **Type Handling**: Automatic conversion for LinkedList and other complex types
- **Error Handling**: Comprehensive error tracking and reporting

## 📊 Evaluation Metrics

### Two Types of Accuracy Measurement

#### 1. **Full Accuracy** 
- Tests against ALL test cases
- Measures complete solution correctness

#### 2. **Partial Accuracy**
- Tests against visible test cases only (excludes last test)
- Measures performance on training examples

#### 3. **Hidden Test Accuracy**
- Tests against only the hidden test case
- Measures **generalization ability**

### Example Output:
```
📊 RESULTS:
Full Tests (All): 3/3 (100.0%)
Partial Tests (Visible): 2/2 (100.0%)
Hidden Test: 1/1 (100.0%)
```

## 🚀 Usage Examples

### Single Problem Evaluation
```python
from automated_code_evaluator import CodeEvaluator

evaluator = CodeEvaluator(model="ollama:gemma3:1b")
result = evaluator.evaluate_problem(problem_data, print_details=True)

print(f"Full Accuracy: {result['results']['full_accuracy']:.1%}")
print(f"Generalization: {result['results']['hidden_accuracy']:.1%}")
```

### Batch Evaluation
```python
from batch_evaluator import BatchEvaluator

batch_evaluator = BatchEvaluator()
results = batch_evaluator.run_all_problems(max_problems=5)
batch_evaluator.generate_report()
batch_evaluator.save_results()
```

### Custom Analysis
```python
# Analyze generalization gaps
generalization_gap = [
    r for r in results 
    if r['status'] == 'success' and 
    r['results']['partial_accuracy'] > r['results']['hidden_accuracy']
]
```

## 🔧 Key Features

### ✅ **Reliable Code Extraction**
- Structured prompts with clear formatting requirements
- Regex-based extraction with fallback mechanisms
- Handles various LLM response formats

### ✅ **Robust Testing Framework**
- Supports all problem types in your JSON file
- Automatic LinkedList conversion for complex problems
- Comprehensive error handling and reporting

### ✅ **Generalization Testing**
- Hides last test case from LLM
- Measures true generalization ability
- Identifies overfitting to visible examples

### ✅ **Comprehensive Reporting**
- Detailed accuracy metrics
- Error analysis and categorization
- Exportable results in JSON format

### ✅ **Flexible Model Support**
- Works with any AISuite-supported model
- Easy model switching for comparisons
- Configurable system prompts

## 📁 File Structure

```
├── aisuite_client.py           # Core AISuite wrapper
├── automated_code_evaluator.py # Main evaluation engine
├── batch_evaluator.py          # Batch processing system
├── evaluation_demo.py          # Usage demonstrations
├── usage_example.py            # Simple usage example
├── leetcode_problems.json      # Problem dataset
└── evaluation_results_*.json   # Generated results
```

## 🎯 Benefits of This Approach

1. **Accuracy**: Reliable code extraction and execution
2. **Generalization**: Tests true problem-solving ability, not memorization
3. **Scalability**: Batch processing for large-scale evaluation
4. **Flexibility**: Works with different models and problem types
5. **Insights**: Detailed error analysis and performance metrics

## 🔮 Future Enhancements

- Support for more complex input/output types
- Integration with more LLM providers
- Advanced prompt engineering techniques
- Performance optimization for large-scale evaluation
- Web interface for interactive evaluation

---

**This system successfully combines the reliability of structured prompts with the power of comprehensive automated testing, providing accurate evaluation of LLM coding capabilities with proper generalization measurement.** 