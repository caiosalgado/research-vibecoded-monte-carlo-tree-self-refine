# 🧬 Research VibeCoded: Monte Carlo Tree Self-Refine

**Part of the Research VibeCoded Series** - Implementing key Artificial Intelligence research papers through hands-on coding projects.

## 📄 Research Paper

This project implements concepts from:

**"Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLama-3 8B: A Technical Report"**  
*Zhang et al., 2024*

## ⚠️ Project Status

**🚧 WORK IN PROGRESS 🚧**

This project is currently in development and implements only the foundational evaluation system. The core Monte Carlo Tree Self-refine algorithm from the paper is not yet implemented.

### Current Implementation:
- ✅ **Automated Code Evaluation System** - Tests LLM-generated solutions
- ✅ **Multi-run Benchmarking** - Statistical reliability testing
- ✅ **Structured Problem Loading** - LeetCode-style coding challenges
- ✅ **Comprehensive Reporting** - CSV and JSON output formats

### Planned Implementation:
- 🚧 **Monte Carlo Tree Search** - Core algorithm from the paper
- 🚧 **Self-refine Mechanism** - Iterative solution improvement
- 🚧 **Performance Optimization** - Matching paper benchmarks

## 🎯 About Research VibeCoded

Research VibeCoded is a series of projects that implement cutting-edge AI research papers through practical, hands-on coding. Each project breaks down complex academic concepts into understandable, executable code.

**Goals:**
- Make AI research accessible through code
- Bridge theory and practice
- Create reusable implementations
- Foster learning and experimentation

## 📁 Current Project Structure

```
├── main.py                    # Interactive single-problem evaluation
├── benchmark_challenges.py    # Multi-model benchmark runner
├── src/
│   ├── __init__.py
│   ├── client.py             # AISuite LLM integration
│   ├── evaluator.py          # Code evaluation engine
│   └── batch_runner.py       # Batch processing & benchmarking
├── data/
│   └── leetcode_problems.json # Problem dataset
└── results/                  # Generated evaluation results
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
uv install

# Ensure Ollama is running with desired models
ollama serve
```

### Single Problem Evaluation

```bash
python main.py
```

### Multi-Model Benchmarking

```bash
python benchmark_challenges.py
```

This will run comprehensive benchmarks across multiple models with 3 runs per problem for statistical reliability.

## 📊 Current Evaluation System

### Features

- **Dual Evaluation**: Partial (visible tests) vs Full (including hidden tests)
- **Multi-Run Testing**: 3 runs per problem for consistency analysis
- **Complex Type Support**: LinkedList, arrays, custom data structures
- **Comprehensive Reporting**: CSV and JSON outputs with timestamps
- **Model Comparison**: Easy switching between different LLM models

### Supported Models

Currently tested with:
- `ollama:gemma3:1b`
- `ollama:deepseek-r1:1.5b`  
- `ollama:qwen3:1.7b`

### Evaluation Metrics

- **Accuracy**: Percentage of test cases passed
- **Consistency Score**: Stability across multiple runs
- **Error Classification**: Runtime errors vs wrong answers
- **Performance Tracking**: Individual and average results

## 🔧 Usage Examples

### Programmatic Usage

```python
from src.batch_runner import run_simple_benchmark

# Run benchmark for specific model
csv_file, json_file = run_simple_benchmark("ollama:qwen3:1.7b")
```

### Custom Problem Evaluation

```python
from src.evaluator import get_problem, CodeTester
from src.client import AISuiteClient

# Load problem and evaluate
problem = get_problem("twoSum")
client = AISuiteClient(model="ollama:gemma3:1b")
tester = CodeTester()

# Your evaluation logic here...
```

## 📈 Research Implementation Roadmap

### Phase 1: Foundation (✅ Current)
- [x] Basic evaluation system
- [x] Multi-model benchmarking
- [x] Statistical analysis framework

### Phase 2: Core Algorithm (🚧 Next)
- [ ] Monte Carlo Tree Search implementation
- [ ] Self-refine mechanism
- [ ] Solution tree exploration
- [ ] Candidate generation and selection

## 🤝 Contributing

This is part of an educational research series. Contributions, suggestions, and discussions about the implementation are welcome!

**Areas for contribution:**
- Core MCTS algorithm implementation
- Additional evaluation metrics
- New problem datasets
- Performance optimizations

## 📚 References

Zhang, et al. (2024). "Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLama-3 8B: A Technical Report."

---

**🧬 Research VibeCoded - Making AI Research Accessible Through Code** 