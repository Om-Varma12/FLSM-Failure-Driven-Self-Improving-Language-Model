def generate_correction(skill, question):
    if skill == "math":
        expr = question.split("What is")[1].replace("?", "")
        return str(eval(expr))

    if skill == "factual":
        return "Narendra Modi is the Prime Minister of India."

    if skill == "reasoning":
        return "Milo is an animal."

    if skill == "coding":
        return "def reverse_string(s): return s[::-1]"

    return "Correct answer unavailable."
