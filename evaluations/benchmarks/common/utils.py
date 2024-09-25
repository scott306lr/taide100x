import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np


def get_logger(
    benchmark_name: str, log_file_path: str = None, logging_level=logging.INFO
):
    logger = logging.getLogger(benchmark_name)
    if not logger.handlers:  # Check if handlers have already been added
        logger.setLevel(logging_level)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file_path is None:
            logfile_name = f"benchmark_{benchmark_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
            log_file_path = os.path.join(os.getcwd(), "logs", logfile_name)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def launch_cli(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--prompt",
        type=str,
        help="The prompt for the model.",
    )
    parser.add_argument("--max_tokens", type=int, help="The maximum number of tokens.")

    parser.add_argument(
        "--repetitions",
        type=int,
        help="The number of repetitions for the benchmark.",
    )
    parser.add_argument(
        "--device",
        help="Device to use for the benchmark.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Path to the models directory.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature to use.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for the benchmark (default: 1).",
    )
    parser.add_argument(
        "--prefill_seq_len",
        type=int,
        default=128,
        help="Prefill sequence length for the benchmark (default: 128).",
    )
    return parser


def make_report(
    args, benchmark_class, runner_dict, benchmark_name, is_bench_pytorch: bool = False
):
    experiment_name = f"{benchmark_name}-{str(datetime.now())}"
    report = defaultdict(lambda: defaultdict(float))
    all_answers = {}

    for instance in runner_dict[args.device]:
        model_path, precision = instance["model_path"], instance["precision"]
        benchmark = benchmark_class(
            model_path=model_path,
            model_name=args.model_name,
            benchmark_name=benchmark_name,
            precision=precision,
            device=args.device,
            experiment_name=experiment_name,
        ).load_model_and_tokenizer()

        logger = benchmark.logger

        # First we do benchmarking
        benchmark.benchmark(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            repetitions=args.repetitions,
            temperature=args.temperature,
            batch_size=args.batch_size,
            prefill_seq_len=args.prefill_seq_len
        )

        # Make report for benchmarks
        # Memory seems to be stay the same, so we can take the max of it

        report[f"{args.model_name}-{benchmark_name} (token/sec)"][precision] = {
            "mean": np.mean(benchmark.tps_results),
            "std": np.std(benchmark.tps_results),
            "median": np.median(benchmark.tps_results),
        }
        report[f"{args.model_name}-{benchmark_name} (memory usage)"][precision] = {
            "usage": max(benchmark.memory_usage_results)
        }

        # Second we get the answers
        benchmark.get_answers()
        all_answers[precision] = benchmark.answers

    # Make the final report
    logger.info(f"=> Batch size: {args.batch_size}")
    logger.info(f"=> Prefill sequence length: {args.prefill_seq_len}")
    for framework, quantizations in report.items():
        for quantization, stats in quantizations.items():
            if framework == f"{args.model_name}-{benchmark_name} (memory usage)":
                logger.info(f"{framework}, {quantization}: {stats['usage']} MB")
            else:
                logger.info(
                    f"{framework}, {quantization} (mean): {stats['mean']:.2f} ± {stats['std']:.2f}"
                )
                logger.info(
                    f"{framework}, {quantization} (median): {stats['median']:.2f}"
                )
    # Finally write the quality checks results
    logger.info("Writing the model completion for empirical tests")
    with open(benchmark.answers_json_path, "w", encoding='utf8') as json_file:
        # json.dump(all_answers, json_file)
        json.dump(all_answers, json_file, indent=4, ensure_ascii=False)

        logger.info("Benchmarking Fininshed")
    markdown_content = make_markdown(
        input_json_path=benchmark.answers_json_path, is_bench_pytorch=is_bench_pytorch
    )

    with open(os.path.join(benchmark.log_folder, "quality.md"), "w") as readme_file:
        readme_file.write("\n".join(markdown_content))

    print("README.md has been created with the table.")


def make_markdown(input_json_path: str, is_bench_pytorch: bool = False):
    with open(input_json_path, "r", encoding='utf-8') as file:
        data = json.load(file)

    precisions = list(data.keys())
    markdown_content = []

    # Helper function to create a markdown table row
    def create_row(items):
        return "| " + " | ".join(items) + " |"

    # Build headers based on the mode
    if is_bench_pytorch:
        headers = ["Question"] + precisions
    else:
        headers = ["Question"] + precisions + ["Ground Truth"]

    markdown_content.append(create_row(headers))
    markdown_content.append(create_row(["---"] * len(headers)))

    # Build the Markdown
    for idx, question in enumerate(data[precisions[0]]):
        question_text = question.get("question", "").replace("\n", " ")

        answers = [
            data[precision][idx]["actual"].replace("\n", "<br>")
            for precision in precisions
        ]
        row_items = [question_text] + answers

        if not is_bench_pytorch:
            ground_truths = [
                data[precision][idx]["expected"].replace("\n", "<br>")
                for precision in precisions
            ]
            row_items += ground_truths
        markdown_content.append(create_row(row_items))

    return markdown_content