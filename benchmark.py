import json
import time
import pandas as pd
from tqdm import tqdm
import requests
import re
import os
from datasets import load_dataset

# -------------------------
# Configuration
# -------------------------

MODELS = [
    "gemma:2b",      # Gemma 2B (use q4/q5 quant in Ollama)
    "qwen2:1.5b",    # Qwen2 1.5B
    "tinyllama"      # ~1.1B
]
PROMPT_FILE = "data/prompts.jsonl"
RESULT_FILE = "results_local.csv"
SLEEP_BETWEEN_REQUESTS = 1  # seconds
MAX_TOKENS = 128

# -------------------------
# Helper Functions
# -------------------------

def load_prompts(path):
    """Load prompts from a JSONL file, skip empty lines."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def call_model(model, item):
    """
    Call local Ollama model via REST API.
    Assumes Ollama is running with `ollama serve`.
    """
    url = "http://127.0.0.1:11434/api/generate"
    prompt = f"Question is {item['question']}.\n Options are A: {item['choices'][0]}, B: {item['choices'][1]}, C: {item['choices'][2]}, D: {item['choices'][3]}. \n Provide in the format of \"Option A/B/C/D \" only. Dont output anything else."
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False, 
        "options": {
            # "temperature": 0.2,
            "num_predict": MAX_TOKENS
        }
    }
    start = time.time()
    try:
        r = requests.post(url, json=payload)
        r.raise_for_status()
        output = r.json()["response"].strip()
        metadata = {
            "total_duration": r.json().get("total_duration", None),
            "load_duration": r.json().get("load_duration", None),
            "prompt_eval_duration": r.json().get("prompt_eval_duration", None),
            "prompt_eval_count": r.json().get("prompt_eval_count", None),
            "eval_count": r.json().get("eval_count", None),
            "eval_duration": r.json().get("eval_duration", None)
        }
        # print(f"Model output: {output}")
    except Exception as e:
        output = f"ERROR: {e}"
        metadata = {
            "total_duration": None,
            "load_duration": None,
            "prompt_eval_duration": None,
            "prompt_eval_count": None,
            "eval_count": None,
            "eval_duration": None
        }
    latency = time.time() - start
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    return output, latency, metadata

def judge_answer(expected_output, ans):
    match = re.search(r':\s*([A-Z])\b', ans, re.IGNORECASE)
    ans = match.group(1) if match else "E"
    ans = ans.strip().upper()
    mapping = { 0: "A", 1: "B", 2: "C", 3: "D" }
    expected_output = mapping[expected_output]
    if expected_output == ans:
        return 1
    else:
        return 0
    

def get_mmlu():
    ds = load_dataset("cais/mmlu", "abstract_algebra")  
    return ds

# -------------------------
# Main Benchmark Loop
# -------------------------

def main():
    mmlu_dataset = get_mmlu()
    test_data = mmlu_dataset["test"]
    test_data = test_data.select(range(2))  # Limit to first 100 for quick testing
    # print(test_data[0])
    results = []

    for model in MODELS:
        print(f"\nRunning model: {model}")
        for id, item in tqdm(enumerate(test_data)):
            output, latency, metadata = call_model(model, item)
            score = judge_answer(item["answer"], output)
            results.append({
                "model": model,
                "prompt_id": id,
                "prompt": item["question"],
                "output": output,
                "score": score,
                "latency_sec": round(latency, 2),
                "metadata": metadata
            })

    df = pd.DataFrame(results)
    df.to_csv(RESULT_FILE, index=False)
    print(f"\nBenchmark complete. Results saved to {RESULT_FILE}")

# -------------------------
# Run Script
# -------------------------

if __name__ == "__main__":
    main()

