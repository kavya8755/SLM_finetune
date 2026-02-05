# Small Language Model Fine-Tuning

Python project for fine-tuning a pretrained causal language model from Hugging Face on a text dataset. Uses the Hugging Face `transformers` + `datasets` stack with the standard `Trainer` API.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run a small example (defaults shown):

```bash
python src/train.py \
  --model_id gpt2 \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --output_dir outputs/gpt2-wikitext
```

Key options:
- `--model_id`: any causal LM on the Hub (e.g., `sshleifer/tiny-gpt2` for smoke tests).
- `--dataset_name`, `--dataset_config`: dataset and config from the Hub.
- `--text_field`: dataset text column (default `text`).
- `--block_size`: sequence length for training (default 128).
- `--max_train_samples` / `--max_eval_samples`: subset for quick runs.

Outputs (checkpoints + tokenizer) land in `--output_dir`. You can inspect logs in the same directory (`trainer_state.json`, `events.out.tfevents.*`).

## Tips

- For laptops, start with `sshleifer/tiny-gpt2` and small batch sizes.
- To use Apple Silicon GPUs, set `PYTORCH_ENABLE_MPS_FALLBACK=1` if kernels fall back to CPU.
- Add `--fp16` on CUDA machines or `--bf16` on BF16-capable GPUs for speed.
- Replace the dataset with your own by pointing to a local JSON/text file via `datasets` (see their docs) and set `--text_field` accordingly.
