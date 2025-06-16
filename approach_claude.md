
## **📋 APPROACH 1: Structured Prompt with Delimiters**

**Best for: Reliability and consistency**

```python
def create_structured_prompt(problem_data):
    prompt = f"""Solve this problem and provide ONLY the function:

{problem_data['description']}

**CRITICAL:** Respond in EXACTLY this format:
```python
def {problem_data['id']}(...):
    # your solution
    return result
```

No explanations, no test cases, just the function."""

def extract_code_delimiters(response):
    pattern = r'```python\s*(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    return matches[0].strip() if matches else ""
```

**Pros:** High extraction success rate, consistent format  
**Cons:** Less natural for LLM, might reduce code quality


## **📋 APPROACH 4: Complete Testing Framework**

**Best for: Production use with comprehensive evaluation**

```python
class AutomatedTester:
    def __init__(self):
        self.namespace = {
            'List': List, 'Optional': Optional,
            'ListNode': self.create_listnode_class()
        }
    
    def create_listnode_class(self):
        class ListNode:
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next
                
            def to_list(self):
                result = []
                current = self
                while current:
                    result.append(current.val)
                    current = current.next
                return result
        return ListNode
    
    def run_evaluation(self, code, problem_data, test_type='all'):
        """
        Args:
            test_type: 'all' or 'partial' (excludes last test)
        """
        tests = problem_data['tests']
        if test_type == 'partial':
            tests = tests[:-1]
        
        results = {'total': len(tests), 'passed': 0, 'accuracy': 0.0}
        
        try:
            exec(code, self.namespace)
            func = self.namespace[problem_data['id']]
            
            for test in tests:
                inputs = self.prepare_inputs(test['input'], problem_data['id'])
                expected = test['expected']
                
                try:
                    actual = func(*inputs) if len(inputs) > 1 else func(inputs[0])
                    
                    # Handle LinkedList results
                    if hasattr(actual, 'to_list'):
                        actual = actual.to_list()
                    
                    if actual == expected:
                        results['passed'] += 1
                        
                except Exception as e:
                    print(f"Test failed: {e}")
                    
        except Exception as e:
            print(f"Execution error: {e}")
            
        results['accuracy'] = results['passed'] / results['total']
        return results
```

## **🚀 RECOMMENDED APPROACH:**

**For your use case, I recommend a hybrid of Approaches 1 and 4:**

1. **Use structured prompts** for reliable code extraction
2. **Implement the complete testing framework** for accurate evaluation
3. **Show partial test cases** in prompt (hide last test for generalization testing)

**Sample Implementation:**

```python
def create_evaluation_prompt(problem_data):
    visible_tests = problem_data['tests'][:-1]  # Hide last test
    
    prompt = f"""Solve this coding problem:

**Problem:** {problem_data['title']}
**Description:** {problem_data['description']}

**Visible Test Cases:**
{visible_tests}

**Note:** There is 1 additional hidden test case.

**IMPORTANT:** Provide your solution in this exact format:
```python
{problem_data['function_signature']}
    # Your implementation here
    pass
```

Only provide the function, no explanations."""
    
    return prompt

# Usage
tester = AutomatedTester()
prompt = create_evaluation_prompt(problem_data)
response = client.respond(prompt, print_response=False)
code = extract_code_delimiters(response)

full_accuracy = tester.run_evaluation(code, problem_data, 'all')
partial_accuracy = tester.run_evaluation(code, problem_data, 'partial')

print(f"Full Accuracy: {full_accuracy['accuracy']:.2%}")
print(f"Partial Accuracy: {partial_accuracy['accuracy']:.2%}")
```

**This approach gives you:**
- ✅ Reliable code extraction  
- ✅ Proper handling of LinkedList problems
- ✅ Both full and partial accuracy metrics
- ✅ Robust error handling
- ✅ Generalization testing (hidden test cases)

Would you like me to implement the complete framework as a working script?