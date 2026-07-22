# Model Comparison

## Overall Metrics

| Configuration | Device | Avg Tok/s | Time Q1 | Time Q2 | Time Q3 | Time Q4 | Time Q5 | VRAM (GB) | RAM (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4-bit CUDA | cuda | 14.17 | 10.26s | 12.33s | 12.26s | 2.41s | 1.24s | 2.002 | 1.190 |
| FP16 CUDA | cuda | 23.24 | 5.80s | 8.43s | 8.51s | 1.38s | 0.75s | 5.784 | 2.078 |
| FP16 CPU | cpu | 1.44 | 43.57s | 39.78s | 48.38s | 13.55s | 7.22s | 0.000 | 11.532 |

## Per-Question Comparison

### Q1: Explain what quantization means for a large language model, in 3 sentences.

| Configuration | Time (s) | Output Tokens | Answer |
|---|---:|---:|---|
| 4-bit CUDA | 10.26 | 87 | Quantization is the process of reducing the precision of the weights and activations in a large language model to save memory and improve inference speed. By using fewer bits to represent these values, such as from 32-bit floating point to 8-bit or lower, the model's size decreases significantly while maintaining performance. This technique is particularly useful for deploying models on devices with limited computational resources, making them more efficient and faster to run. |
| FP16 CUDA | 5.80 | 110 | Quantization is the process of reducing the precision of the numbers used to represent the weights and activations in a large language model, typically from floating-point (like 32-bit floats) to lower precision formats like 8-bit integers. This reduction in precision can significantly decrease the model's memory footprint and computational requirements, making it more efficient for deployment on devices with limited resources. However, this process often involves some loss of accuracy, which may be acceptable if the trade-off between performance and resource usage is beneficial for the specific application or use case. |
| FP16 CPU | 43.57 | 32 | Quantization is the process of reducing the precision of the numbers used to represent the weights and activations in a large language model, typically from floating-point (like  |

### Q2: Write a Python function that returns the nth Fibonacci number using memoization.

| Configuration | Time (s) | Output Tokens | Answer |
|---|---:|---:|---|
| 4-bit CUDA | 12.33 | 200 | Certainly! To create a Python function that returns the nth Fibonacci number using memoization, we can use a dictionary to store previously computed values. This will help us avoid redundant calculations and make the function more efficient.<br><br>Here's how you can implement it:<br><br>```python<br>def fibonacci(n, memo={}):<br>    # Base cases<br>    if n == 0:<br>        return 0<br>    elif n == 1:<br>        return 1<br>    <br>    # Check if the value is already computed<br>    if n in memo:<br>        return memo[n]<br>    <br>    # Compute the value and store it in the memo dictionary<br>    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)<br>    <br>    return memo[n]<br><br># Example usage:<br>n = 10<br>print(f"The {n}th Fibonacci number is: {fibonacci(n)}")<br>```<br><br>### Explanation:<br>- The `memo` dictionary is used to store previously computed Fibonacci numbers.<br>- |
| FP16 CUDA | 8.43 | 200 | Certainly! To compute the nth Fibonacci number efficiently using memoization, we can store previously computed values in a dictionary to avoid redundant calculations. Here's how you can implement this:<br><br>```python<br>def fibonacci(n, memo={}):<br>    # Base cases: if n is 0 or 1, return n directly<br>    if n <= 1:<br>        return n<br>    <br>    # Check if the value is already computed and stored in the memo dictionary<br>    if n not in memo:<br>        # If not, compute it recursively and store it in the memo dictionary<br>        memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)<br>    <br>    return memo[n]<br><br># Example usage:<br>n = 10  # Change this value to get the nth Fibonacci number<br>print(f"The {n}th Fibonacci number is: {fibonacci(n)}")<br>```<br><br>### Explanation:<br>- **Base Cases**: The first two Fibonacci numbers are defined as `0` |
| FP16 CPU | 39.78 | 32 | Certainly! To compute the nth Fibonacci number efficiently using memoization, we can store previously computed values in a dictionary to avoid redundant calculations. Here's how you can |

### Q3: A store sells apples in bags of 6 and oranges in bags of 4. I need exactly 24 apples and 20 oranges. How many bags of each do I need? Show your reasoning.

| Configuration | Time (s) | Output Tokens | Answer |
|---|---:|---:|---|
| 4-bit CUDA | 12.26 | 200 | To determine the number of bags needed for 24 apples and 野20 oranges, we can use the given information about the bag sizes.<br><br>For apples:<br>- Each bag contains 6 apples.<br>- We need a total of 24 apples.<br><br>To find out how many bags of apples are needed, we divide the total number of apples by the number of apples per bag:<br><br>\[ \text{Number of apple bags} = \frac{\text{Total number of apples}}{\text{Apples per bag}} = \frac{24}{6} = 4 \]<br><br>So, we need 4 bags of apples.<br><br>For oranges:<br>- Each bag contains 4 oranges.<br>- We need a total of 20 oranges.<br><br>To find out how many bags of oranges are needed, we divide the total number of oranges by the number of oranges per bag:<br><br>\[ \text{Number of orange bags} = \frac{\text{Total number of oranges}}{\text |
| FP16 CUDA | 8.51 | 200 | To determine how many bags of apples and oranges you need to buy to get exactly 24 apples and 20 oranges, we can set up the problem using the given quantities per bag.<br><br>1. **Apples:**<br>   - Each bag contains 6 apples.<br>   - You need 24 apples.<br>   - To find out how many bags of apples you need, divide the total number of apples needed by the number of apples per bag:<br>     \[<br>     \frac{24 \text{ apples}}{6 \text{ apples/bag}} = 4 \text{ bags}<br>     \]<br><br>2. **Oranges:**<br>   - Each bag contains 4 oranges.<br>   - You need 20 oranges.<br>   - To find out how many bags of oranges you need, divide the total number of oranges needed by the number of oranges per bag:<br>     \[<br>     \frac{20 \text{ oranges}}{4 \text{ oranges/b |
| FP16 CPU | 48.38 | 32 | To determine how many bags of apples and oranges you need to buy to get exactly 24 apples and 20 oranges, we can set up the problem |

### Q4: Summarize the following in one sentence: 'The Transformer architecture, introduced in 2017, replaced recurrence with self-attention, allowing models to process sequences in parallel and scale far more efficiently than RNNs.'

| Configuration | Time (s) | Output Tokens | Answer |
|---|---:|---:|---|
| 4-bit CUDA | 2.41 | 35 | The Transformer architecture, introduced in 2017, replaced recurrence with self-attention, enabling parallel processing of sequences and significantly improving efficiency compared to RNNs. |
| FP16 CUDA | 1.38 | 34 | The Transformer architecture, introduced in 2017, replaced recurrence with self-attention to enable parallel sequence processing and significantly improve efficiency compared to RNNs. |
| FP16 CPU | 13.55 | 32 | The Transformer architecture, introduced in 2017, replaced recurrence with self-attention to enable parallel sequence processing and significantly improve efficiency compared to RNNs |

### Q5: Write a two-line rhyming couplet about debugging code at 2am.

| Configuration | Time (s) | Output Tokens | Answer |
|---|---:|---:|---|
| 4-bit CUDA | 1.24 | 19 | At midnight, the debugger shines,<br>Fixing bugs that won't go on their own. |
| FP16 CUDA | 0.75 | 19 | At midnight, the debugger's light,<br>Fixes bugs that won't play at night. |
| FP16 CPU | 7.22 | 19 | At midnight, the debugger's light,<br>Fixes bugs that won't play at night. |
