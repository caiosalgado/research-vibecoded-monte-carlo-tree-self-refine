# 🎯 Automated Code Evaluation System

Clean, organized implementation for evaluating LLM-generated code solutions.

## 📁 Project Structure

```
├── main.py                    # Main evaluation script (follows your pseudocode)
├── batch_test.py             # Batch evaluation example
├── aisuite_client.py         # AISuite wrapper (from previous work)
├── src/
│   ├── __init__.py
│   ├── core.py               # Essential functions
│   └── batch_runner.py       # Batch processing utilities
├── data/
│   └── leetcode_problems.json # Problem dataset
└── results/                  # Generated evaluation results
```

## 🚀 Quick Start

### Single Problem Evaluation

Run the main script that follows your exact pseudocode structure:

```bash
uv run python main.py
```

The script uses `# %%` markers for clear step separation:

1. **Get Problem** - Load problem by ID
2. **Create Input Prompt** - Generate structured prompt  
3. **LLM Response** - Get AI solution
4. **Extract Code** - Parse code from response
5. **Run Evaluations** - Test with partial and full test cases
6. **Save Results** - Export evaluation data

### Batch Evaluation

Test multiple problems at once:

```bash
uv run python batch_test.py
```

### Customize Problem

Edit `main.py` to change the problem:

```python
# %% Get Problem
problem_id = "isPalindrome"  # Change this to test different problems
```

Available problem IDs:
- `isPalindrome`
- `romanToInt` 
- `twoSum`
- `addTwoNumbers`
- `longestPalindrome`
- `maxArea`
- `mergeKLists`
- `findMedianSortedArrays`
- `isMatch`

## 📊 Evaluation Metrics

The system provides two accuracy measurements:

1. **Partial Accuracy**: Tests against visible test cases only
2. **Full Accuracy**: Tests against all test cases (including hidden ones)

This allows measuring both training performance and generalization ability.

## 🔧 Core Functions

### Essential Functions (`src/core.py`):

- `get_problem(problem_id)` - Load problem data
- `create_evaluation_prompt(problem_data)` - Generate structured prompt
- `extract_code_delimiters(response)` - Parse code from LLM response
- `CodeTester.run_evaluation(code, problem, test_type)` - Execute and test code

### Evaluation Structure:

```python
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
```

## 🎨 Features

✅ **Step-by-step execution** with `# %%` markers  
✅ **Clean code extraction** using regex patterns  
✅ **Generalization testing** with hidden test cases  
✅ **Complex type support** (LinkedList, etc.)  
✅ **Comprehensive error handling**  
✅ **Organized file structure**  
✅ **Batch processing capabilities**  

## 💡 Usage Tips

1. **IDE Support**: The `# %%` markers work great with VS Code's Python Interactive window
2. **Custom Models**: Change the model in `main.py` or `batch_test.py`
3. **Result Analysis**: Check the `results/` directory for detailed JSON output
4. **Error Debugging**: Full error details are included in the evaluation results

## 🔧 Requirements

- Python 3.8+
- AISuite library
- Ollama with desired models
- uv package manager

---

**Clean, focused, and ready for your evaluation needs! 🚀** 