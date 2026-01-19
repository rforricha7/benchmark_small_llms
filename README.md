# benchmark_small_llms

Benchmarks for small LLMs focusing on latency and quality evaluation. This repository provides simple scripts to measure latency, token usage, and evaluation duration across models and prompts, plus plotting helpers to visualize results.

## What this project does

- Runs benchmark prompts (downloaded from huggingface) against multiple models.
- Measures latency, prompt/output token counts, and evaluation duration.
- Saves raw results to `results_local.csv` and generates plots in `plots/`.
- Provides small test and example scripts for quick validation.

## Repo structure

- `benchmark.py` - main benchmarking script (runs prompts and records metrics).
- `benchmark_test.py` - an example / test harness for the benchmark logic.
- `chart.py`, `latency_chart.py` - plotting utilities used to create visualizations in `plots/`.
- `results_local.csv` - example output from a previous run.
- `plots/` - folder containing generated PNG charts (token counts vs latency/duration, etc.).
- `requirements.txt` - Python dependencies for running the project.
- `test.py` - small quick-run script or smoke test.

## Quickstart

Requirements:
- Python 3.10+ recommended (the project was developed with modern Python).
- A working Python environment and pip.

1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Inspect or customise prompts

The prompts are stored in `data/prompts.jsonl`. Each line is a single JSON object representing one prompt/test case.

3) Run a benchmark

The repository includes `benchmark.py`. Depending on how your models are configured, run it like:

```bash
# run the main benchmark (may require env vars or local model endpoints)
python benchmark.py

# run the example/test harness
python benchmark_test.py

# run a quick smoke test
python test.py
```

Notes:
- Some scripts may expect environment variables or a model endpoint. Check the top of the script for required configuration and API keys.
- `results_local.csv` will be overwritten/updated by runs and contains the raw per-prompt metrics.

## Outputs and plots

- After a run, plots are created/updated in the `plots/` directory. Example filenames include:
	- `prompt_tokens_vs_latency.png`
	- `output_tokens_vs_latency.png`
	- `output_tokens_vs_eval_duration.png`
	- `total_tokens_vs_latency.png`
- `avg_score_by_model.png` is an example summary plot (if evaluation scoring is implemented and available).

## Extending the benchmark

- Add or change prompts in `data/prompts.jsonl`.
- Modify `benchmark.py` to point to different model endpoints or change sampling/parameters.
- Use the plotting helpers in `chart.py` and `latency_chart.py` to create custom visualizations.

## Tests and validation

- `benchmark_test.py` contains quick checks of the benchmark functions. Run it with:

```bash
python benchmark_test.py
```

If you add tests, follow the pattern used in the existing test files and keep them lightweight so they run quickly.

## Typical development workflow

1. Create/activate virtualenv and install deps
2. Edit prompts or benchmarking parameters
3. Run `python benchmark.py` to collect results
4. Open plots in `plots/` or inspect `results_local.csv`

## Troubleshooting

- If plots don't appear, ensure the plotting libraries from `requirements.txt` are installed and that `plots/` is writable.
- If a script needs credentials for a model API, set the appropriate environment variables (consult the top of the Python script for exact names).

## Contributing

Contributions are welcome. For small fixes, open a PR with a clear description of the change and a short test or example demonstrating it.

## License

See the `LICENSE` file in this repository for license terms.

## Contact / Notes

If you want help adding support for a new model or integrating this benchmark into a CI pipeline, open an issue with the details and a sample model endpoint.

Happy benchmarking!
