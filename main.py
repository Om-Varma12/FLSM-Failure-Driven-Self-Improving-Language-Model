import config
import random

from models.student import student_generate
from evaluation.judge import judge_answer, generate_correction
from tasks.task_generator import generate_task
from memory.memory_store import store_failure
from training.trainer import fine_tune


def main():
    for iteration in range(config.ITERATIONS):

        print(f"\n=== ITERATION {iteration} ===")

        # choose skill
        skill = random.choice(config.SKILLS)
        print("Skill:", skill)

        # generate question
        question = generate_task(skill)
        print("Q:", question)

        # student answer
        student_answer = student_generate(question)
        print("Student:", student_answer)

        # teacher judges
        correct = judge_answer(question, student_answer)

        if correct:
            print("✅ Correct")
            continue

        print("❌ Wrong")

        # teacher correction
        corrected = generate_correction(question)
        print("Teacher correction:", corrected)

        # store failure
        store_failure(question, student_answer, corrected, skill)

        # periodic training
        if iteration % config.TRAIN_EVERY == 0 and iteration > 0:
            print("🔧 Fine-tuning student...")
            fine_tune()

    print("Done.")


if __name__ == "__main__":
    main()