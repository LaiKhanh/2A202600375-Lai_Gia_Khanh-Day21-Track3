# Lab 21 — Evaluation Report

**Học viên**: Lại Gia Khánh — 2A202600375 <br>
**Ngày nộp**: 2026-05-07 <br>
**Submission option**: A (lightweight)

## 1. Setup
- **Base model**: unsloth/Llama-3.2-3B-Instruct-bnb-4bit
- **Dataset**: unsloth/Llama-3.2-3B-Instruct-bnb-4bit, 200 samples (180 train + 20 eval)
- **max_seq_length**: 512 (p95 = 300, rounded up)
- **GPU**: Tesla T4, 15.6 GB VRAM
- **Training cost**: $0.05 (~8.7 phút @ $0.35/hr)


## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | 2293760             | 2.779907 min    | 5.639620 GB    | 0.880123       | 2.411196        |
| 16   | 4587520             | 3.031979 min    | 4.875621 GB    | 0.863137       | 2.370585        |
| 64   | 18350080             | 2.861985 min    | 6.617683 GB    | 0.844456       | 2.326711        |
| Base | -               | -          | -         | 2.326711       | 2.37        |

## 3. Loss Curve Analysis
![LOSS_CURVE_IMG](./loss_curve.png)
- Quan sát: Không bị overfitting vì Train loss giảm khá đều từ khoảng 1.33 xuống ~0.78–0.88, Eval loss ở cuối khoảng ~0.86, khá gần với train loss, khoảng cách giữa train và eval loss không lớn, nên model chưa học quá mức dữ liệu train.

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Write a Python function to generate the first n Fibonacci numbers using iteration. <br>
**Base**: 
```python
def fibonacci(n):
    fib_sequence = [0, 1]
    while len(fib_sequence) < n:
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence[:n]

print(fibonacci(10))  # Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
print(fibonacci(5))   # Output: [0, 1, 1, 2, 3]
print(fibona)
```

**Fine-tuned (r=16)**: 
```python
def generate_fibonacci(n):
    fib_sequence = [0, 1]
    while len(fib_sequence) < n:
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence[:n] 

# Example usage
print(generate_fibonacci(10)) # prints [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]  # The first 10 Fibonacci numb
```
**Nhận xét**: Improved

### Example 2
**Prompt**: Create a Python function that checks whether a given integer is prime or not. <br>
**Base**:
```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

# Test the function
print(is_prime(5))  # Returns: True
print(is_prime(10))  # Returns: False
print(is_prime(11))  # Returns: True
print(is_prime(24))  # Ret
```
**Fine-tuned (r=16)**:
```python
# Testing the function
print(is_prime(11))  # Returns: True
print(is_prime(10))  # Returns: False
print(is_prime(7))   # Returns: True
print(is_prime(4))   # Returns: False
print(is_prime(13))  # Returns: True
print(is_prime(0))   # Returns: False
print(is_prime(1))   # Returns: False
print(is_prime)
```
**Nhận xét**: Same

### Example 3
**Prompt**: Write a Python program that reverses the order of words in a sentence without using built-in reverse functions. <br>
**Base**:
```python
def reverse_words(sentence):
    words = sentence.split()
    reversed_words = []
    for word in words:
        reversed_word = ""
        for i in range(len(word)):
            reversed_word = word[len(word)-i-1] + reversed_word
        reversed_words.append(reversed_word)
    return " ".join(reve
```
**Fine-tuned (r=16)**:
```python
def reverse_words(sentence):
    words = sentence.split(' ')
    reversed_words = []
    for word in words:
        reversed_word = ''
        for char in word:
            reversed_word = char + reversed_word
        reversed_words.append(reversed_word)
    return''.join(reversed_words)

sentence =
```
**Nhận xét**: Degraded

### Example 4
**Prompt**: Write Python code to read a CSV file and calculate the average value of a numeric column named salary. <br>
**Base**:
```python
import pandas as pd

# Read the CSV file
df = pd.read_csv('salary.csv')

# Calculate the average salary
average_salary = df['salary'].mean()

print("Average salary:", average_salary)
```
**Fine-tuned (r=16)**:
```python
import pandas as pd

def calculate_average_salary(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Calculate the average salary
    average_salary = df['salary'].mean()

    return average_salary

csv_file = 'data.csv'
average_salary = calculate_average_salary(csv_file)
print
```
**Nhận xét**: Improved

### Example 5
**Prompt**: Create a Python script that loads a JSON file and pretty-prints it with indentation. <br>
**Base**:
```python
import json

# Load the JSON file
with open('data.json', 'r') as f:
    data = json.load(f)

# Pretty-print the data
print(json.dumps(data, indent=4)) 

# Output:
# {
#     "key1": "value1",
#     "key2": "value2",
#     "key3": "value3"
# } 
# 
# Note: 
# - The indent parameter specifies the number
```
**Fine-tuned (r=16)**:
```python
import json

with open('data.json', 'r') as file:
    data = json.load(file)

print(json.dumps(data, indent=4)) 

# Output:
# {
#     "key1": "value1",
#     "key2": "value2"
# } 
# 
# Response:
# import json

# with open('data.json', 'r') as file:
#     data = json.load(file)

# print(json.dumps(da
```
**Nhận xét**: Same

## 5. Conclusion về Rank Trade-off

- Trong thí nghiệm này, rank r=16 cho ROI tốt nhất trên dataset gồm 200 samples. So với r=8, model đạt perplexity thấp hơn và output ổn định hơn trong khi thời gian train và VRAM usage chỉ tăng nhẹ. Khi tăng lên r=64, perplexity tiếp tục giảm nhưng mức cải thiện khá nhỏ so với lượng trainable parameters tăng rất lớn. Điều này cho thấy với dataset nhỏ, rank quá cao không tận dụng được hết capacity của adapter và dễ dẫn đến lãng phí tài nguyên.
- Diminishing returns bắt đầu xuất hiện từ khoảng r=16 → r=64. Mặc dù r=64 có kết quả tốt nhất về metric, chênh lệch perplexity không đáng kể so với r=16, trong khi số parameters tăng khoảng 4 lần. Ngoài ra, qualitative examples cũng cho thấy chất lượng output không cải thiện tương ứng với mức tăng resource usage.
- Nếu deploy production cho bài toán lightweight hoặc inference cost-sensitive, tôi sẽ chọn r=16. Rank này đạt sự cân bằng tốt giữa quality, tốc độ train, VRAM usage và chi phí triển khai. Với dataset nhỏ và domain đơn giản như Python instruction tuning, r=16 đủ để model học adaptation cần thiết mà không làm mô hình trở nên quá nặng.

## 6. What I Learned
- LoRA rank ảnh hưởng trực tiếp đến số trainable parameters, VRAM usage và chất lượng fine-tuning.
- Eval loss và perplexity không phải lúc nào cũng phản ánh rõ chất lượng output thực tế, nên cần kết hợp qualitative evaluation.
- Với dataset nhỏ, tăng rank quá cao thường cho diminishing returns và không cải thiện đáng kể hiệu năng model.