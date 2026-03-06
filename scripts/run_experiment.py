"""Launch the full FSLM pipeline from the command line."""
import argparse

from pipeline.full_pipeline import run_full_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run the Failure-Driven Self-Improving Language Model pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to the config YAML file",
    )
    args = parser.parse_args()

    run_full_pipeline(args.config)


if __name__ == "__main__":
    main()
