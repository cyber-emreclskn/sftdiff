"""An intentional objective change using a real HF tokenizer and collator.

Install optional dependencies with `pip install '.[hf]'` before running.
This is a local integration smoke test, NOT a reproduction of an upstream bug.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from sftdiff import capture_batch, compare_snapshots
from sftdiff.report import render_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="hf-demo")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--cache-dir", default=".hf-cache")
    args = parser.parse_args()
    out = Path(args.out)
    outputs = [out / name for name in ("before.jsonl", "after.jsonl", "report.html", "comparison.json")]
    if any(path.exists() for path in outputs):
        parser.error("Output files already exist; choose a fresh --out directory.")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir, trust_remote_code=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        parser.error("This example requires a tokenizer with a defined pad token.")
    samples = [
        ("service-001", "Yapılan işlem: filtre temizlendi.", "Filtre temizliği yapıldı."),
        ("order-002", "Customer ordered two boxes of bolts.", "Quantity: 2. Unit: box. Product: bolts."),
        ("tool-003", "Return the stock count as JSON: forty-two.", '{"stock":42}'),
    ]
    encoded, prompt_lengths, row_ids = [], [], []
    for row_id, prompt, answer in samples:
        history = [{"role": "user", "content": prompt}]
        prefix = tokenizer.apply_chat_template(history, tokenize=True, add_generation_prompt=True, return_dict=False)
        full = tokenizer.apply_chat_template(history + [{"role": "assistant", "content": answer}],
                                             tokenize=True, add_generation_prompt=False, return_dict=False)
        if full[:len(prefix)] != prefix:
            parser.error("Template is not prefix-stable; this example cannot infer the response boundary.")
        encoded.append({"input_ids": full, "attention_mask": [1] * len(full)})
        prompt_lengths.append(len(prefix))
        row_ids.append(row_id)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="np")
    all_tokens = collator(encoded)
    response_only = {key: value.copy() for key, value in all_tokens.items()}
    response_only["labels"] = np.full_like(all_tokens["input_ids"], -100)
    for i, (prefix_len, source) in enumerate(zip(prompt_lengths, encoded)):
        response_only["labels"][i, prefix_len:len(source["input_ids"])] = source["input_ids"][prefix_len:]
    metadata = {"packing": False, "pipeline": "hf_example.py", "intentional_change": True,
                "transformers": importlib.metadata.version("transformers"),
                "numpy": np.__version__, "template_sha256": hashlib.sha256(
                    tokenizer.chat_template.encode("utf-8")).hexdigest()}
    out.mkdir(parents=True, exist_ok=True)
    before = capture_batch(outputs[0], response_only, tokenizer, row_ids=row_ids,
                           metadata={**metadata, "objective": "response-only"})
    after = capture_batch(outputs[1], all_tokens, tokenizer, row_ids=row_ids,
                          metadata={**metadata, "objective": "all non-padding tokens"})
    result = compare_snapshots(before, after)
    outputs[2].write_text(render_report(result, before_name="response-only", after_name="all-token objective"), encoding="utf-8")
    outputs[3].write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print("Intentional objective-change report:", outputs[2])


if __name__ == "__main__":
    main()
