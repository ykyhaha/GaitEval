"""LoRA SFT for Qwen2.5-7B-Instruct using completion-only label loss."""
import os
import json
import argparse

def load_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data-dir", required=True, help="dir containing the prepared train.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="enable gradient checkpointing (needed for 72B; PEFT-safe)")
    ap.add_argument("--no-merge", action="store_true",
                    help="skip LoRA merge (serve base+adapter via vLLM --enable-lora; for 72B)")
    ap.add_argument("--device-map", default=None,
                    help="None for single-GPU (7B); 'auto' to shard a big model (32B/72B)")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = load_jsonl(os.path.join(args.data_dir, "train.jsonl"))
    if not rows:
        raise ValueError("Training partition is empty")
    for row in rows:
        messages = row["messages"]
        if (len(messages) != 2 or messages[0]["role"] != "user"
                or messages[1]["role"] != "assistant" or not messages[1]["content"].strip()):
            raise ValueError("Each training record needs one prompt and one label completion")
    texts = [tok.apply_chat_template(r["messages"], tokenize=False) for r in rows]
    if any(len(tok(text, add_special_tokens=False)["input_ids"]) > args.max_seq_len
           for text in texts):
        raise ValueError("A training example exceeds --max-seq-len; label truncation is not allowed")
    ds = Dataset.from_dict({"text": texts})
    print(f"Training examples: {len(ds)}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map=(args.device_map or None),
    )
    if args.grad_ckpt:
        # PEFT + gradient checkpointing needs inputs to require grad explicitly.
        model.enable_input_require_grads()
    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM", bias="none",
    )
    # Completion-only: supervise ONLY the assistant (label) tokens.
    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n", tokenizer=tok)

    cfg = SFTConfig(
        output_dir=os.path.join(args.out_dir, "ckpt"),
        per_device_train_batch_size=args.bsz, gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_ratio=0.03, bf16=True, logging_steps=20, save_strategy="no",
        max_seq_length=args.max_seq_len, dataset_text_field="text", packing=False,
        report_to="none", gradient_checkpointing=args.grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.grad_ckpt else None,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         peft_config=peft_cfg, tokenizer=tok, data_collator=collator)
    trainer.train()

    adapter_dir = os.path.join(args.out_dir, "adapter")
    trainer.save_model(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"[INFO] saved adapter -> {adapter_dir}")

    if args.no_merge:
        print("[INFO] --no-merge: skipping merge; serve base + adapter via vLLM --enable-lora")
        return
    # Merge LoRA into base and save a standalone model for vLLM eval.
    merged = trainer.model.merge_and_unload()
    merged_dir = os.path.join(args.out_dir, "merged")
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)
    print(f"[INFO] saved merged model -> {merged_dir}")


if __name__ == "__main__":
    main()
