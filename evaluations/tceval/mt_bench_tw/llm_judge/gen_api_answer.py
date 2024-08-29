"""Generate answers with GPT-4

# NOTE: code adapted from fastchat/llm_judge/gen_api_answer.py

Usage:
python3 gen_api_answer.py --model gpt-3.5-turbo
"""
import argparse
import json
import os
import time
import concurrent.futures
from typing import Optional

import openai
import shortuuid
import tqdm

from common import (
    load_questions,
    temperature_config,
    chat_completion_openai
)

from conversation import get_conv_template

# Adapted from fastchat.llm_judge.gen_model_answer.reorg_answer_file
def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r", encoding="utf-8") as fin:
        for l in fin:
            data = json.loads(l)
            qid = data["question_id"]
            answers[qid] = data

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w", encoding="utf-8") as fout:
        for qid in qids:
            fout.write(json.dumps(answers[qid], ensure_ascii=False) + "\n")

# Apply reference
def apply_ref(question, ref):
    if len(ref) == 0:
        return question
    template = f"You are given a question and its reference answer.\nQuestion: {question}\nReference: {ref}\n\n\Take the reference into account and answer the question. Think step by step if needed."
    return template


def get_answer(
    question: dict, 
    model: str, 
    num_choices: int, 
    max_tokens: int, 
    answer_file: str, 
    sys_mesg: str = None,
    use_ref: bool = False
):
    assert (
        args.force_temperature is not None and "required_temperature" in question.keys()
    ) == False
    if args.force_temperature is not None:
        temperature = args.force_temperature
    elif "required_temperature" in question.keys():
        temperature = question["required_temperature"]
    elif question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]
    else:
        temperature = 0.7
    print(f".... temperature = {temperature}")
    choices = []
    chat_state = None  # for palm-2 model
    for i in range(num_choices):
        conv = get_conv_template("raw")
        conv.set_system_message("")
        if sys_mesg is not None:
            conv.set_system_message(sys_mesg)

        turns = []
        for j in range(len(question["turns"])):
            # Apply reference to question
            usr_q = question["turns"][j]
            if use_ref and "reference" in question:
                usr_q = apply_ref(usr_q, question['reference'][j])
            conv.append_message(conv.roles[0], usr_q)
            conv.append_message(conv.roles[1], None)

            output = chat_completion_openai(model, conv, temperature, max_tokens)

            conv.update_last_message(output)
            turns.append(output)

        choices.append({"index": i, "turns": turns})

    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": choices,
        "tstamp": time.time(),
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(ans, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench_tw",
        help="The name of the benchmark question set.",
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--model", type=str, default="aide-gpt-35-turbo-16k-4k")
    parser.add_argument("--model-sys-mesg", type=str, default=None, help="System mesg for the model")
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--force-temperature", type=float, help="Forcibly set a sampling temperature."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument("--openai-api-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--use-ref", type=bool, default=False, help="(debug) Use the reference in the question to cheat the answer generation")
    args = parser.parse_args()

    if args.openai_api_base is not None:
        openai.api_base = args.openai_api_base

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.question_file is not None:
        print(f".... found question file. Use {args.question_file} instead of default {question_file}")
        question_file = args.question_file

    questions = load_questions(question_file, args.question_begin, args.question_end)

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model}.jsonl"
    print(f"Output to {answer_file}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for question in questions:
            future = executor.submit(
                get_answer,
                question,
                args.model,
                args.num_choices,
                args.max_tokens,
                answer_file,
                args.model_sys_mesg,
                args.use_ref
            )
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    reorg_answer_file(answer_file)
