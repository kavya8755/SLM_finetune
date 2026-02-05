import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import evaluate
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass
class DataConfig:
    dataset_name: str
    dataset_config: Optional[str]
    text_field: str
    block_size: int
    max_train_samples: Optional[int]
    max_eval_samples: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM with Hugging Face Trainer")
    parser.add_argument("--model_id", type=str, default="gpt2", help="Pretrained model repo id on Hugging Face Hub")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Dataset name on Hugging Face Hub")
    parser.add_argument(
        "--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset config on the Hub (if applicable)"
    )
    parser.add_argument("--text_field", type=str, default="text", help="Column containing raw text")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to store checkpoints")
    parser.add_argument("--block_size", type=int, default=128, help="Sequence length after tokenization")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Train batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Eval batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit training samples for smoke tests")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Limit eval samples for smoke tests")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 if available (CUDA)")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 if available")
    return parser.parse_args()


def load_and_prepare_dataset(cfg: DataConfig, tokenizer: AutoTokenizer):
    raw_ds = load_dataset(cfg.dataset_name, cfg.dataset_config)

    def tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(batch[cfg.text_field])

    tokenized = raw_ds.map(
        tokenize,
        batched=True,
        remove_columns=[col for col in raw_ds["train"].column_names if col != cfg.text_field],
    )

    def group_texts(examples: Dict[str, List[List[int]]]) -> Dict[str, Any]:
        concatenated: List[int] = sum(examples["input_ids"], [])
        total_length = (len(concatenated) // cfg.block_size) * cfg.block_size
        concatenated = concatenated[:total_length]
        result = {
            "input_ids": [concatenated[i : i + cfg.block_size] for i in range(0, total_length, cfg.block_size)],
        }
        # Labels are the same as input_ids for causal LM.
        result["labels"] = result["input_ids"].copy()
        return result

    lm_ds = tokenized.map(group_texts, batched=True)

    if cfg.max_train_samples:
        lm_ds["train"] = lm_ds["train"].select(range(cfg.max_train_samples))
    if cfg.max_eval_samples and "validation" in lm_ds:
        lm_ds["validation"] = lm_ds["validation"].select(range(cfg.max_eval_samples))

    return lm_ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Shift for causal LM perplexity calculation
    shift_logits = torch.tensor(logits[:, :-1, :])
    shift_labels = torch.tensor(labels[:, 1:])
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
    perplexity = torch.exp(loss)
    return {"perplexity": perplexity.item()}


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id)

    data_cfg = DataConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config if args.dataset_config else None,
        text_field=args.text_field,
        block_size=args.block_size,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    dataset = load_and_prepare_dataset(data_cfg, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if "validation" in dataset else None,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
