import argparse
import importlib.util
from pathlib import Path


def load_traj_module():
    module_path = Path(__file__).resolve().parent / "trajectory_generation.py"
    spec = importlib.util.spec_from_file_location("traj", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description="Compute accuracy for an existing trajectory JSONL file.")
    parser.add_argument(
        "--input",
        type=str,
        default="/data1/home/yunhochoi/verl/data/math_variant_valid_llama_trajectories_4.jsonl",
        help="Path to generated trajectory JSONL (must include 'trajectory' and 'answer' fields).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="existing",
        help="Label for the printed report.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    mod = load_traj_module()
    metrics = mod.compute_accuracy_from_file(input_path)
    mod.print_accuracy_report(metrics, args.label)


if __name__ == "__main__":
    main()
