import random

def generate_task(skill):
    if skill == "math":
        q = random.randint(10, 50)
        return f"What is {q} * 3?"

    if skill == "reasoning":
        return "If all cats are animals and Milo is a cat, what is Milo?"

    if skill == "coding":
        return "Write a Python function to reverse a string."

    if skill == "factual":
        return "Who is the Prime Minister of India?"

    return "Explain gravity."