from models.teacher import teacher_generate

def judge_answer(question, student_answer):
    prompt = f"""
You are a strict evaluator.

Question: {question}
Student Answer: {student_answer}

Is the student correct?
Reply ONLY with YES or NO.
"""
    result = teacher_generate(prompt).lower()
    return "yes" in result


def generate_correction(question):
    prompt = f"""
Give the correct answer to this question:

{question}
"""
    return teacher_generate(prompt)
