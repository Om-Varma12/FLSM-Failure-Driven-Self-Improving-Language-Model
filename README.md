# FSLM — Failure-Driven Self-Improving Language Model

FSLM is a tool that **automatically teaches a small AI model to get better at a task by learning from its own mistakes**.

You pick a skill (like math), run the pipeline, and the model improves itself over multiple rounds — no manual labeling needed.

---

## How It Works

1. The **student model** (a small AI) tries to answer questions from a dataset
2. Its wrong answers are collected into a **failure memory bank**
3. A **teacher model** (a powerful AI via the Groq API) analyzes each mistake and generates corrected examples
4. The student is fine-tuned on those corrections
5. This cycle repeats — each round, the student makes fewer mistakes

---

## What You Need Before Starting

- **Python 3.10 or higher**
- **A GPU** (recommended — training runs on GPU; CPU is very slow)
- **A Groq API key** — the teacher model runs on [Groq](https://console.groq.com). Sign up for a free account and copy your API key.

---

## Step 1 — Install Dependencies

Clone the repo and install all required packages:

```bash
git clone <repo-url>
cd FSLM-Failure-Driven-Self-Improving-Language-Model
pip install -r requirements.txt
```

---

## Step 2 — Set Up Your API Key

Create a file named `.env` in the root of the project folder and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

> You can get your key from [https://console.groq.com/keys](https://console.groq.com/keys)

---

## Step 3 — Prepare the Dataset

Download and prepare the math dataset (GSM8K) that the model will train and be tested on:

```bash
python scripts/prepare_dataset.py
```

This saves 300 training questions and 200 test questions to `data/processed/math/`.

---

## Step 4 — Run Baseline Evaluation (Optional)

Before any training, you can check how well the student model already performs:

```bash
python scripts/run_baseline.py
```

This prints the model's starting accuracy. A sample of its failures is saved to `outputs/results/baseline_failures.json` so you can see where it struggles.

---

## Step 5 — Run the Full Training Pipeline

This is the main step. It runs the full failure-driven improvement loop:

```bash
python scripts/run_experiment.py
```

Or with a custom config file:

```bash
python scripts/run_experiment.py --config configs/base_config.yaml
```

The pipeline will:
- Evaluate the student model before training (baseline)
- Run 3 improvement iterations by default
- After each iteration, save a checkpoint to `outputs/checkpoints/`
- Print a results table showing accuracy improvements
- Save a plot of the accuracy curve to `outputs/plots/`

Example output:

| Stage       | Accuracy |
| ----------- | -------- |
| Baseline    | 26%      |
| Iteration 1 | 39%      |
| Iteration 2 | 48%      |
| Iteration 3 | 56%      |

---

## Step 6 — Making the Model Even Better

To squeeze out more improvement, edit `configs/base_config.yaml`:

| Setting | What It Does | How to Improve |
|---|---|---|
| `num_iterations` | Number of training rounds | Increase to `5` or more |
| `num_epochs` | Training passes per round | Try `5` for deeper learning |
| `learning_rate` | How fast the model learns | Lower to `1e-4` if unstable |
| `batch_size` | Samples per training step | Lower to `2` if you run out of GPU memory |

After editing, re-run:

```bash
python scripts/run_experiment.py
```

---

## Where Results Are Saved

| Path | Contents |
|---|---|
| `outputs/results/` | Accuracy results as JSON |
| `outputs/plots/` | Accuracy improvement chart |
| `outputs/checkpoints/` | Model checkpoints after each iteration |

---

## Supported Skills

Currently the pipeline is set up for **math reasoning** using the GSM8K dataset.

To switch skills, update the `skill` field in `configs/base_config.yaml`. New skills can be added under `configs/skills/`.

---

## Summary

```
Install deps → Add API key → Prepare data → Run pipeline → Check results → Tune config → Re-run
```