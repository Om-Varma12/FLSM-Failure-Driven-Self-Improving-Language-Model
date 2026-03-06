skill-forge/                          ← root project name (suggestion)
│
├── configs/                          ← all YAML/JSON config files
│   ├── base_config.yaml              ← global settings (models, paths, hyperparams)
│   ├── skills/
│   │   ├── math.yaml                 ← per-skill config (dataset path, eval metric, prompts)
│   │   ├── sql.yaml
│   │   ├── code_debug.yaml
│   │   └── ...
│   └── lora_config.yaml              ← LoRA hyperparams (r, alpha, target modules)
│
├── data/                             ← all data lives here
│   ├── raw/                          ← original downloaded datasets
│   │   ├── math/
│   │   ├── sql/
│   │   └── ...
│   ├── processed/                    ← cleaned, formatted, split datasets
│   │   ├── math/
│   │   │   ├── train.jsonl
│   │   │   └── test.jsonl
│   │   └── ...
│   └── failure_bank/                 ← failure memory bank (per skill, per iteration)
│       ├── math/
│       │   ├── iter_1.jsonl
│       │   ├── iter_2.jsonl
│       │   └── iter_3.jsonl
│       └── ...
│
├── skills/                           ← skill plugin definitions
│   ├── base_skill.py                 ← abstract base class for all skills
│   ├── math_skill.py                 ← implements input/output/eval for math
│   ├── sql_skill.py
│   ├── code_debug_skill.py
│   └── registry.py                   ← skill registry (name → class mapping)
│
├── student/                          ← student model code
│   ├── loader.py                     ← load model + tokenizer (BnB 4-bit + unsloth)
│   ├── inference.py                  ← run student on batches of inputs
│   └── lora_manager.py              ← save/load/merge LoRA adapters per iteration
│
├── teacher/                          ← teacher model interface
│   ├── client.py                     ← API client for llama-3.3-70b (Groq/Together/etc.)
│   ├── analyzer.py                   ← send failure → get error analysis + correction
│   ├── dataset_generator.py          ← convert teacher analysis → training samples
│   └── test_case_generator.py       ← generate harder/edge-case problems
│
├── training/                         ← fine-tuning logic
│   ├── trainer.py                    ← main LoRA fine-tuning loop (uses HF Trainer / Unsloth)
│   ├── data_collator.py             ← batch formatting for training
│   └── curriculum.py                ← curriculum-aware failure ordering (easy→hard)
│
├── evaluation/                       ← evaluation and metrics
│   ├── evaluator.py                  ← run student on test set, compute accuracy
│   ├── metrics.py                    ← skill-specific metrics (exact match, BLEU, exec accuracy)
│   └── reporter.py                   ← generate iteration-wise accuracy tables/plots
│
├── pipeline/                         ← orchestration / main loop
│   ├── run_iteration.py              ← single iteration: attempt → fail → analyze → finetune
│   ├── full_pipeline.py             ← runs N iterations end-to-end
│   └── checkpoint_manager.py        ← save/restore state between iterations
│
├── prompts/                          ← all LLM prompt templates
│   ├── teacher_analysis.txt          ← prompt: given failure, analyze and correct
│   ├── teacher_datagen.txt           ← prompt: convert failure to training sample
│   ├── teacher_testgen.txt           ← prompt: generate harder test cases
│   └── student_inference.txt         ← system prompt for student during inference
│
├── outputs/                          ← experiment outputs (gitignored except structure)
│   ├── checkpoints/                  ← LoRA adapter weights per iteration
│   │   └── math/
│   │       ├── iter_1/
│   │       ├── iter_2/
│   │       └── iter_3/
│   ├── results/                      ← accuracy logs, JSON result files
│   └── plots/                        ← accuracy curves, error distribution charts
│
├── notebooks/                        ← exploratory Jupyter notebooks
│   ├── 01_baseline_eval.ipynb
│   ├── 02_failure_analysis.ipynb
│   └── 03_results_visualization.ipynb
│
├── tests/                            ← unit tests
│   ├── test_skill_registry.py
│   ├── test_teacher_client.py
│   └── test_evaluator.py
│
├── scripts/                          ← one-off CLI scripts
│   ├── prepare_dataset.py            ← download + preprocess raw data
│   ├── run_baseline.py               ← evaluate before any training
│   └── run_experiment.py             ← launch full pipeline from CLI
│
├── requirements.txt
├── README.md
└── .env                              ← API keys (GROQ_API_KEY, etc.) — gitignored