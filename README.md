# Project: Teacher-Guided Failure-Driven Skill Acquisition for Small Language Models

## Core Idea

The goal of the project is to **teach specific skills to a small language model (SLM)** using a **teacher model and an iterative failure-driven training loop**.

Instead of training on a static dataset, the system:

1. Evaluates the student model on tasks
2. Detects where it fails
3. Stores failures in a memory bank
4. Uses a stronger teacher model to analyze those failures
5. Generates corrected reasoning and training examples
6. Fine-tunes the student model
7. Repeats the process

The key concept is:

```
learning from mistakes
```

This mimics how humans learn from teachers.

---

# Models Used

## Student Model (SLM)

Student model:

```
unsloth/Llama-3.2-3B-Instruct-bnb-4bit
```

Purpose:

This is the model whose **skill we want to improve**.

Why small model?

* cheaper training
* fits into consumer GPUs
* improvements are easier to measure

The model will be loaded with **4-bit quantization using** **BitsAndBytes**.

Why 4-bit?

Normally model weights are loaded as:

* FP32 (32-bit)
* FP16 (16-bit)

With 4-bit quantization:

```
16 bit → 4 bit
```

Memory usage becomes roughly:

```
25% of the FP16 model
```

This makes it possible to train on smaller GPUs.

---

## Teacher Model (LLM)

Teacher model:

```
llama-3.3-70b-versatile
```

The teacher model is **much stronger than the student**.

Responsibilities of the teacher:

* detect mistakes in student outputs
* explain why the student failed
* generate corrected reasoning
* create new training examples
* generate harder test cases

So the teacher acts as:

```
teacher = evaluator + critic + dataset generator
```

---

# What Is a "Skill"?

A skill is a **specific task capability** that the model can perform.

Example skills:

* mathematical reasoning
* SQL generation
* code debugging
* logical reasoning
* information extraction
* tool usage
* classification
* structured output generation

A skill can be represented as:

```
Input → transformation → output
```

Example:

Math reasoning

```
Question → reasoning steps → answer
```

Example:

Code debugging

```
Buggy code → bug detection → fixed code
```

---

# Skill-Based Training Concept

The student model starts with **baseline capability**.

Then we improve **one skill at a time**.

The training process is:

```
baseline → failure detection → correction → fine-tuning → improvement
```

This is done **iteratively**.

---

# Dataset for Each Skill

Each skill requires a dataset.

Example structure:

```
Input
Expected Output
(Optional) Reasoning Steps
```

Example (math):

```
Question:
Tom has 3 apples and buys 4 more. How many apples does he have?

Answer:
7
```

Example (code debugging):

```
Buggy code:
print(i)

Correct code:
for i in range(10):
    print(i)
```

Datasets can come from:

1. public datasets
2. synthetic generation
3. failure-based generation

---

# Dataset Split

Example experiment setup:

```
Total samples: 500

Train: 300
Test: 200
```

Train set → learning
Test set → evaluation

---

# Baseline Evaluation

First, we evaluate the student model **before training**.

Steps:

1. Send test questions to the student model
2. Collect outputs
3. Compare outputs with correct answers

Accuracy formula:

```
accuracy = correct answers / total questions
```

Example:

```
correct = 52
total = 200

accuracy = 26%
```

This becomes the **baseline performance**.

---

# Training Pipeline

Now the main pipeline begins.

---

# Step 1: Student Attempts Training Problems

The student model attempts problems from the training dataset.

For each sample:

```
student_output = student(input)
```

---

# Step 2: Compare with Gold Answer

Each output is compared with the correct answer.

```
if student_output == gold_answer
    mark as success
else
    mark as failure
```

Gold answer means:

```
correct answer provided in the dataset
```

---

# Step 3: Store Failure

If the student fails, the failure is stored in the **failure memory bank**.

Example entry:

```json
{
 "input": "...",
 "student_output": "...",
 "gold_output": "...",
 "error_type": "...",
 "error_reason": "...",
 "corrected_solution": "...",
 "correct_reasoning": "..."
}
```

This memory bank becomes a **record of all mistakes**.

---

# Why We Need Failure Memory Bank

The memory bank enables:

### Targeted learning

Instead of retraining on everything, we train only on **mistakes**.

### Error pattern discovery

Example distribution:

```
logic errors = 40
calculation errors = 25
missing step = 15
format errors = 10
```

This shows **model weaknesses**.

### Iterative learning

Failures can be revisited in later training iterations.

---

# Teacher Analysis

When a failure is detected, the teacher model receives:

```
input
student output
gold answer
```

The teacher performs:

```
error detection
error explanation
correct reasoning generation
correct answer generation
```

Example teacher output:

```
Error: missing step in reasoning

Explanation:
The student added numbers incorrectly.

Correct reasoning:
3 apples + 4 apples = 7 apples

Correct answer:
7
```

---

# Dataset Generation from Failures

The teacher converts failures into **training samples**.

Example format:

```
Input
Correct reasoning
Correct answer
```

These samples are stored as **new training data**.

This is called **failure-driven dataset generation**.

---

# Fine-Tuning the Student

Once enough failure samples are collected, we fine-tune the student model.

Fine-tuning method:

**LoRA** introduced by **Microsoft**.

Advantages:

* fewer trainable parameters
* faster training
* lower memory usage

---

# Iterative Learning Loop

Training does not happen once.

The process runs in cycles.

```
iteration 1
student attempts tasks
failures stored
fine-tune student

iteration 2
student attempts again
new failures stored
fine-tune student

iteration 3
repeat
```

Each iteration reduces mistakes.

---

# Teacher-Generated Test Cases

To test the limits of the model, the teacher can generate **new problems**.

Example:

```
create problems that require multi-step reasoning
create harder variations
create edge cases
```

These help test whether the student **truly learned the skill**.

---

# Final Evaluation

After training, we run the **test dataset again**.

Example results:

| Stage       | Accuracy |
| ----------- | -------- |
| Baseline    | 26%      |
| Iteration 1 | 39%      |
| Iteration 2 | 48%      |
| Iteration 3 | 56%      |

Improvement indicates **successful skill acquisition**.

---

# Multi-Skill Expansion

Once the framework works for one skill, we can extend it.

Example domains:

```
math reasoning
code debugging
SQL generation
logical reasoning
tool usage
```

Each skill uses the **same training loop**.

---

# Final System Architecture

```
Dataset
   ↓
Student model
   ↓
Predictions
   ↓
Failure detection
   ↓
Failure memory bank
   ↓
Teacher analysis
   ↓
Dataset generation
   ↓
Fine-tuning
   ↓
Repeat
```

---

# Final Outcome

The system becomes a **general framework for teaching skills to small language models** using teacher-guided failure analysis and iterative learning.

This allows small models to **progressively acquire complex capabilities** without requiring massive training datasets.