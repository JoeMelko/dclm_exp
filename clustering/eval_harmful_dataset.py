#!/usr/bin/env python
"""
eval_harmful_dataset.py
-----------------------
Compute the token-level cross-entropy loss (and perplexity) of an Open-LM
checkpoint – identified by its Datacomp-LM run UUID – on the HuggingFace
`LLM-LAT/harmful-dataset`.

Each dataset record contains a *prompt* along with a *rejected* and (usually)
*accepted* completion.  For evaluation we feed the model the concatenation

    "<prompt><rejected>"

and compute the loss **only** over the *rejected* portion (the prompt tokens
are masked out with label -100).

Example
-------
python eval_harmful_dataset.py --uuid <RUN_UUID>

Optional arguments
------------------
--split         Dataset split to evaluate (default: "train")
--batch-size    Evaluation batch size (default: 16)
--max-length    Truncate sequences to this many tokens (default: 2048)
"""
# --------------------------------------------------------------------------- #
import argparse, math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch, tqdm
from torch.cuda.amp import autocast
from transformers import AutoTokenizer

# --------------------------------------------------------------------------- #
#                        Optional dependency: datasets                        #
# --------------------------------------------------------------------------- #
try:
    from datasets import load_dataset  # type: ignore
except ImportError as e:  # nocov
    raise ImportError("The 'datasets' package is required – install via `pip install datasets`."
                      ) from e

# Enable TF-32 on modern NVIDIA GPUs – safe speedup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --------------------------------------------------------------------------- #
#                      Model loader (reuse from eval_openhermes)              #
# --------------------------------------------------------------------------- #
# We reuse the helper defined in eval_openhermes.py.  Support both package and
# standalone execution.
try:
    from .eval_openhermes import load_openlm_model_from_uuid  # type: ignore
except ImportError:  # nocov
    import sys as _sys
    _this_dir = Path(__file__).resolve().parent
    if str(_this_dir) not in _sys.path:
        _sys.path.insert(0, str(_this_dir))
    from eval_openhermes import load_openlm_model_from_uuid  # type: ignore

# --------------------------------------------------------------------------- #
#                          Data processing helpers                            #
# --------------------------------------------------------------------------- #

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

# ----------------------------- dataset wrappers ---------------------------- #
class HarmfulDataset(torch.utils.data.Dataset):
    """HuggingFace `LLM-LAT/harmful-dataset` – use *rejected* as completion."""

    NAME = "harmful-dataset"

    def __init__(self, split: str = "train"):
        self.ds = load_dataset("LLM-LAT/harmful-dataset", split=split)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        rec = self.ds[idx]
        return {"prompt": rec["prompt"], "completion": rec["rejected"]}


class AdvBenchDataset(torch.utils.data.Dataset):
    """HuggingFace `walledai/AdvBench` – use *target* as completion."""

    NAME = "advbench"

    def __init__(self, split: str = "train"):
        # `AdvBench` is single split "train" in HF hub; ignore split argument.
        self.ds = load_dataset("walledai/AdvBench", split="train")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        rec = self.ds[idx]
        return {"prompt": rec["prompt"], "completion": rec["target"]}

# --------------------------------------------------------------------------- #
#                             Dataset registry                                #
# --------------------------------------------------------------------------- #

DATASET_REGISTRY: dict[str, type[torch.utils.data.Dataset]] = {
    "harmful": HarmfulDataset,
    "advbench": AdvBenchDataset,
}

# -------------------------- collate / padding ------------------------------ #

def collate(batch, *, max_length: int = 2048):
    """Tokenise and pad a batch.

    1. For each sample compute tokens of *prompt* and *prompt+rejected*.
    2. Build *labels* masking the prompt tokens with -100 so the loss is only
       evaluated on the rejected portion.
    3. Pad tokens to the length of the longest sequence in the batch; pad labels
       with -100 so they are ignored by the loss.
    """
    toks, labels = [], []
    pad_id = tokenizer.eos_token_id

    for sample in batch:
        prompt_str   = sample["prompt"]
        completion_str = sample["completion"]

        prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        text_ids   = tokenizer(prompt_str + completion_str, add_special_tokens=False)["input_ids"]

        # Truncate if necessary (keep at least full prompt so mask aligns)
        if len(text_ids) > max_length:
            text_ids = text_ids[:max_length]
            # If prompt was truncated resize prompt_ids accordingly
            if len(prompt_ids) > max_length:
                prompt_ids = prompt_ids[:max_length]
        prompt_len = len(prompt_ids)

        # Build labels: -100 over prompt, token id over rejected portion
        lbl = [-100] * prompt_len + text_ids[prompt_len:]
        # If truncation removed part of rejected, align lbl length
        lbl = lbl[: len(text_ids)]

        toks.append(np.asarray(text_ids, dtype=np.int32))
        labels.append(np.asarray(lbl,      dtype=np.int32))

    # Pad to equal length
    L = max(len(x) for x in toks)
    pad_toks   = lambda x: np.pad(x, (0, L - len(x)), constant_values=pad_id)
    pad_labels = lambda x: np.pad(x, (0, L - len(x)), constant_values=-100)

    toks_tensor   = torch.tensor([pad_toks(t)   for t in toks],   dtype=torch.long)
    labels_tensor = torch.tensor([pad_labels(l) for l in labels], dtype=torch.long)
    return toks_tensor, labels_tensor

# --------------------------------------------------------------------------- #
#                               Evaluation                                    #
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def evaluate(model: torch.nn.Module,
             loader: torch.utils.data.DataLoader,
             device: torch.device,
             total_samples: int | None = None) -> float:
    total_loss, total_tokens = 0.0, 0
    tmp_loss, tmp_tokens = 0.0, 0
    ct = 0
    total_batches = math.ceil(total_samples / loader.batch_size) if total_samples else None
    for toks, labels in tqdm.tqdm(loader, desc="Evaluating", total=total_batches):
        toks, labels = toks.to(device), labels.to(device)
        with autocast(dtype=torch.bfloat16):
            out = model(input_ids=toks, labels=labels)
        n_tokens       = (labels != -100).sum().item()
        total_loss   += out.loss.item() * n_tokens
        total_tokens += n_tokens
        tmp_loss += out.loss.item() * n_tokens
        tmp_tokens += n_tokens
        ct += 1
        if ct % 500 == 0:
            print(tmp_loss / tmp_tokens)
            tmp_loss, tmp_tokens = 0.0, 0
    return total_loss / total_tokens

# --------------------------------------------------------------------------- #
#                                   Main                                      #
# --------------------------------------------------------------------------- #

def main(args):
    # ----- model ---------------------------------------------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_openlm_model_from_uuid(args.uuid)
    model.to(device, dtype=torch.bfloat16).eval()

    # ----- dataset -------------------------------------------------------- #
    # Determine which datasets to evaluate.
    selected_names = args.dataset or list(DATASET_REGISTRY.keys())

    for name in selected_names:
        ds_cls = DATASET_REGISTRY.get(name)
        if ds_cls is None:
            raise ValueError(f"Unsupported dataset: {name}")

        print(f"\n=== Evaluating dataset: {name} ===")
        ds = ds_cls(split=args.split)
        loader = torch.utils.data.DataLoader(ds,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=8,
                                             collate_fn=lambda x: collate(x, max_length=args.max_length))

        # ----- evaluation ------------------------------------------------- #
        total_samples = len(ds)
        loss = evaluate(model, loader, device, total_samples=total_samples)
        print(f"Dataset: {name}\n  Cross-entropy loss (completion only): {loss:.6f}\n  Perplexity                         : {math.exp(loss):.6f}\n")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uuid", required=True,
                        help="Datacomp-LM / Open-LM run UUID identifying the checkpoint to evaluate")
    parser.add_argument("--dataset", choices=list(DATASET_REGISTRY.keys()), nargs="*", default=[],
                        help="Datasets to evaluate (space separated). If omitted, all datasets are evaluated.")
    parser.add_argument("--split", default="train",
                        help="Dataset split to evaluate (default: train)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Evaluation batch size (default: 16)")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum sequence length in tokens (default: 2048)")
    main(parser.parse_args()) 