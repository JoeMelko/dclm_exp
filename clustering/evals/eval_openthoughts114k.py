#!/usr/bin/env python
"""
eval_openthoughts114k.py
------------------------
Compute the token-level cross-entropy loss (and perplexity) of an Open-LM
checkpoint – identified by its Datacomp-LM run UUID – on the
OpenThoughts-114k reasoning dataset.

Each sample builds a prompt from an optional system message and the conversation
history up to (but not including) the last assistant message. We then score the
model only on the last assistant response tokens (prompt positions are masked).

Dataset is automatically retrieved from the HuggingFace dataset
"open-thoughts/OpenThoughts-114k" if not already present locally.

Example
-------
python eval_openthoughts114k.py --uuid <RUN_UUID>

Optional arguments
------------------
--batch-size      Evaluation batch size (default: 8)
--max-length      Maximum sequence length after tokenization (default: 2048)
--split           Dataset split to evaluate (default: train)
--total-samples   Limit the number of samples for quick checks (default: None)
"""
# --------------------------------------------------------------------------- #
import argparse, math
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import torch, tqdm
from transformers import AutoTokenizer
from torch.cuda.amp import autocast

# --------------------------------------------------------------------------- #
#                     Optional dependency: datasets                           #
# --------------------------------------------------------------------------- #
try:
    from datasets import load_dataset
except ImportError:  # nocov
    load_dataset = None

# Enable TF-32 for safe speedups
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# --------------------------------------------------------------------------- #
#                         Model loader (reuse from eval_openhermes)           #
# --------------------------------------------------------------------------- #
# When executed within a package (e.g. `python -m dclm_exp.clustering.eval_openthoughts114k`) the
# relative import below works.  When run as a standalone script, the package
# context is absent, so we fall back to adding this file's directory to
# `sys.path` and performing an absolute import instead.
try:
    from .eval_openhermes import load_openlm_model_from_uuid  # type: ignore
except ImportError:  # nocov
    import sys as _sys
    from pathlib import Path as _Path
    _this_dir = _Path(__file__).resolve().parent
    if str(_this_dir) not in _sys.path:
        _sys.path.insert(0, str(_this_dir))
    from eval_openhermes import load_openlm_model_from_uuid  # type: ignore


# --------------------------------------------------------------------------- #
#                         Tokenizer                                           #
# --------------------------------------------------------------------------- #
# Use the GPT-NeoX tokenizer to match Open-LM checkpoints as in other evals
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


# --------------------------------------------------------------------------- #
#                         Formatting helpers                                  #
# --------------------------------------------------------------------------- #
@dataclass
class OpenThoughtsSample:
    prompt_text: str
    response_text: str


USER_ALIASES = {"user", "human", "customer", "student"}
ASSISTANT_ALIASES = {"assistant", "gpt", "bot", "model", "teacher"}


def _normalize_role(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    r = raw.lower()
    if r in USER_ALIASES:
        return "user"
    if r in ASSISTANT_ALIASES:
        return "assistant"
    return r


def _extract_messages(conv_obj) -> List[Tuple[str, str]]:
    """Return list of (role, content) pairs from a conversations field.

    Supports common schemas with keys {role, content} or {from, value}.
    Silently drops malformed messages.
    """
    messages: List[Tuple[str, str]] = []
    if not conv_obj:
        return messages
    for item in conv_obj:
        role = item.get("role") if isinstance(item, dict) else None
        if role is None and isinstance(item, dict):
            role = item.get("from") or item.get("speaker") or item.get("author") or item.get("name")
        content = None
        if isinstance(item, dict):
            content = item.get("content")
            if content is None:
                content = item.get("value") or item.get("text") or item.get("message")
        nr = _normalize_role(role)
        if nr and content is not None:
            messages.append((nr, str(content)))
    return messages


def build_prompt_and_response(system_text: str, messages: List[Tuple[str, str]]) -> Optional[OpenThoughtsSample]:
    """Build prompt from system + history up to last assistant, and take that assistant as response.

    If there is no assistant message, returns None.
    """
    last_asst_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i][0] == "assistant":
            last_asst_idx = i
            break
    if last_asst_idx is None:
        return None

    history = messages[:last_asst_idx]
    response_text = messages[last_asst_idx][1]

    parts: List[str] = []
    if system_text and system_text.strip():
        parts.append("System:\n" + system_text.strip() + "\n")
    for role, content in history:
        if role == "user":
            parts.append("User:\n" + content + "\n")
        elif role == "assistant":
            parts.append("Assistant:\n" + content + "\n")
        else:
            # Unknown roles are included verbatim to preserve context
            parts.append(role.capitalize() + ":\n" + content + "\n")
    parts.append("Assistant:\n")
    prompt_text = "\n".join(parts)
    return OpenThoughtsSample(prompt_text=prompt_text, response_text=response_text)


# --------------------------------------------------------------------------- #
#                         Collation / tokenization                            #
# --------------------------------------------------------------------------- #
@dataclass
class CollatedBatch:
    input_ids: torch.Tensor  # int64 [B, L]
    labels: torch.Tensor     # int64 [B, L]


def _truncate_for_max_length(prompt_ids: List[int], response_ids: List[int], max_length: int) -> Tuple[List[int], List[int]]:
    """Ensure the concatenated sequence fits within max_length.

    Preference: keep the full response if possible; truncate prompt from the left.
    If response alone exceeds max_length, truncate response from the right.
    """
    if len(response_ids) >= max_length:
        return [], response_ids[:max_length]
    available_for_prompt = max_length - len(response_ids)
    if len(prompt_ids) <= available_for_prompt:
        return prompt_ids, response_ids
    return prompt_ids[-available_for_prompt:], response_ids


def collate(batch: List[OpenThoughtsSample], max_length: int) -> CollatedBatch:
    """Tokenize and pad a batch of OpenThoughts samples.

    - Input is prompt + response
    - Labels are -100 for prompt positions, and response token ids for response positions
    - Sequences are padded to max length in the batch with eos as pad and labels -100
    """
    prompt_id_seqs: List[List[int]] = []
    response_id_seqs: List[List[int]] = []

    for s in batch:
        prompt_ids: List[int] = tokenizer.encode(s.prompt_text, add_special_tokens=False)
        response_ids: List[int] = tokenizer.encode(s.response_text or "", add_special_tokens=False) + [tokenizer.eos_token_id]
        p_ids, r_ids = _truncate_for_max_length(prompt_ids, response_ids, max_length)
        if not r_ids:
            # Ensure we keep at least an eos token as target
            r_ids = [tokenizer.eos_token_id]
            if len(p_ids) >= max_length:
                p_ids = p_ids[-(max_length - 1):]
        prompt_id_seqs.append(p_ids)
        response_id_seqs.append(r_ids)

    lengths = [len(p) + len(r) for p, r in zip(prompt_id_seqs, response_id_seqs)]
    L = max(lengths) if lengths else 1

    pad_id = tokenizer.eos_token_id

    input_ids = np.full((len(batch), L), pad_id, dtype=np.int64)
    labels    = np.full((len(batch), L), -100,  dtype=np.int64)

    for i, (p, r) in enumerate(zip(prompt_id_seqs, response_id_seqs)):
        seq = p + r
        input_ids[i, : len(seq)] = np.array(seq, dtype=np.int64)
        # mask prompt with -100, and expose response token ids for loss
        labels[i, len(p) : len(seq)] = np.array(r, dtype=np.int64)

    return CollatedBatch(
        input_ids=torch.from_numpy(input_ids),
        labels=torch.from_numpy(labels),
    )


# --------------------------------------------------------------------------- #
#                               Evaluation                                    #
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def evaluate(model: torch.nn.Module,
             loader: torch.utils.data.DataLoader,
             device: torch.device,
             total_samples: Optional[int] = None) -> float:
    total_loss, total_tokens = 0.0, 0
    tmp_loss, tmp_tokens = 0.0, 0
    ct = 0
    total_batches = math.ceil(total_samples / loader.batch_size) if total_samples else None
    for batch in tqdm.tqdm(loader, desc="Evaluating", total=total_batches):
        toks, labels = batch
        toks, labels = toks.to(device), labels.to(device)
        with autocast(dtype=torch.bfloat16):
            out = model(input_ids=toks, labels=labels)
        n_tokens = (labels != -100).sum().item()
        total_loss += out.loss.item() * n_tokens
        total_tokens += n_tokens
        tmp_loss += out.loss.item() * n_tokens
        tmp_tokens += n_tokens
        ct += 1
        if ct % 500 == 0 and tmp_tokens > 0:
            print(tmp_loss / tmp_tokens)
    if total_tokens == 0:
        return float("nan")
    return total_loss / total_tokens


# --------------------------------------------------------------------------- #
#                           Dataset loader                                     #
# --------------------------------------------------------------------------- #
DATASET_REPO = "open-thoughts/OpenThoughts-114k"


def _ensure_datasets_available():
    if load_dataset is None:
        raise RuntimeError(
            "datasets is not installed – run `pip install datasets` "
            "to enable OpenThoughts-114k evaluation."
        )


def _load_split(split: str, total_samples: Optional[int]) -> List[OpenThoughtsSample]:
    _ensure_datasets_available()
    ds = load_dataset(DATASET_REPO, split=split)
    if total_samples is not None:
        ds = ds.select(range(min(total_samples, len(ds))))

    samples: List[OpenThoughtsSample] = []
    for ex in ds:
        system_text = ex.get("system", "") or ex.get("system_prompt", "")
        conv = ex.get("conversations") or ex.get("messages")
        msgs = _extract_messages(conv)
        built = build_prompt_and_response(system_text, msgs)
        if built is not None:
            samples.append(built)
    return samples


# --------------------------------------------------------------------------- #
#                                   Main                                      #
# --------------------------------------------------------------------------- #

def main(args):
    # ----- model ---------------------------------------------------------- #
    device = torch.device("cuda")

    model = load_openlm_model_from_uuid(args.uuid)
    model.to(device, dtype=torch.bfloat16).eval()

    # ----- dataset -------------------------------------------------------- #
    data = _load_split(args.split, args.total_samples)

    # Build DataLoader with on-the-fly collation
    def _collate_wrapper(samples: List[OpenThoughtsSample]):
        cb = collate(samples, max_length=args.max_length)
        return cb.input_ids, cb.labels

    loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=_collate_wrapper,
        pin_memory=True,
    )

    # ----- evaluation ----------------------------------------------------- #
    eval_total = len(data)
    loss = evaluate(model, loader, device, total_samples=eval_total)
    print(f"Cross-entropy loss: {loss:.6f}\nPerplexity       : {math.exp(loss):.6f}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uuid", required=True,
                        help="Datacomp-LM / Open-LM run UUID identifying the checkpoint to evaluate")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Evaluation batch size (default: 8)")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum sequence length after tokenization (default: 2048)")
    parser.add_argument("--split", default="train",
                        help="Dataset split to evaluate (default: train)")
    parser.add_argument("--total-samples", type=int, default=None,
                        help="Limit the number of samples for a quick evaluation run (default: None)")
    main(parser.parse_args()) 