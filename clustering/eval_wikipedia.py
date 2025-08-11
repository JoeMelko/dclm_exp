#!/usr/bin/env python
"""
Evaluate an Open-LM checkpoint (by Datacomp-LM run UUID) on the
Hugging Face `wikimedia/wikipedia` dataset by computing token-level
cross-entropy loss and perplexity.

Dataset reference: https://huggingface.co/datasets/wikimedia/wikipedia

Example
-------
python -m dclm_exp.clustering.eval_wikipedia --uuid <RUN_UUID> \
    --config 20231101.en --batch-size 8 --block-size 2048 --max-sequences 20000 \
    --shuffle-buffer 10000 --seed 42

Optional arguments
------------------
--config           Dataset configuration like "20231101.en" (default: 20231101.en)
--batch-size       Evaluation batch size (default: 8)
--block-size       Token chunk length for evaluation (default: 2048)
--max-sequences    Max token chunks to evaluate (default: None = no cap)
--num-workers      DataLoader workers (IterableDataset; default: 0)
--shuffle-buffer   Streaming shuffle buffer size (0 disables; default: 0)
--seed             RNG seed for shuffling (default: 0)
"""
# --------------------------------------------------------------------------- #
import argparse, math
from typing import Iterator, List, Optional

import numpy as np
import torch, tqdm
from transformers import AutoTokenizer
from torch.cuda.amp import autocast

# We stream articles with the datasets library
try:
    from datasets import load_dataset  # type: ignore
except Exception as _e:  # nocov
    load_dataset = None  # type: ignore

# Enable TF-32 for safe speedups
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# --------------------------------------------------------------------------- #
#                         Model loader (reuse if run as module)               #
# --------------------------------------------------------------------------- #
# When executed within a package (e.g. `python -m dclm_exp.clustering.eval_wikipedia`) the
# relative import below works. When run as a standalone script, fall back to absolute import.
try:  # type: ignore
    from .eval_openhermes import load_openlm_model_from_uuid  # type: ignore
except ImportError:  # nocov
    import sys as _sys
    from pathlib import Path as _Path
    _this_dir = _Path(__file__).resolve().parent
    if str(_this_dir) not in _sys.path:
        _sys.path.insert(0, str(_this_dir))
    from eval_openhermes import load_openlm_model_from_uuid  # type: ignore


# --------------------------------------------------------------------------- #
#                              Tokenizer                                      #
# --------------------------------------------------------------------------- #
# Use the GPT-NeoX-20B tokenizer, which matches the tokenizer used in other evaluators.
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")


# --------------------------------------------------------------------------- #
#                     IterableDataset over token chunks                       #
# --------------------------------------------------------------------------- #
class WikipediaTokenChunks(torch.utils.data.IterableDataset):
    """Stream `wikimedia/wikipedia` and yield fixed-length token chunks.

    Each yielded element is a 1-D numpy array of token ids with length <= block_size.
    Padding is handled in the collate function.
    """

    def __init__(
        self,
        config: str,
        block_size: int,
        max_sequences: Optional[int] = None,
        shuffle_buffer: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.block_size = int(block_size)
        self.max_sequences = max_sequences
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)

        if load_dataset is None:
            raise RuntimeError(
                "The 'datasets' library is required. Install via `pip install datasets`."
            )

    def _article_iterator(self) -> Iterator[dict]:
        ds = load_dataset("wikimedia/wikipedia", self.config, streaming=True)
        split = ds["train"]  # type: ignore[index]
        if self.shuffle_buffer > 0:
            # Buffer-based shuffle for streaming IterableDataset
            split = split.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)  # type: ignore[attr-defined]
        return iter(split)

    def __iter__(self) -> Iterator[np.ndarray]:
        produced = 0
        for ex in self._article_iterator():
            text = ex.get("text", None)
            if not text:
                continue
            # Tokenize without adding special tokens; chunk into windows
            ids: List[int] = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            if not ids:
                continue
            # Emit chunks of up to block_size tokens
            for start in range(0, len(ids), self.block_size):
                chunk = ids[start : start + self.block_size]
                if len(chunk) < 2:  # need at least one label token
                    continue
                yield np.asarray(chunk, dtype=np.int64)
                produced += 1
                if self.max_sequences is not None and produced >= self.max_sequences:
                    return


# --------------------------------------------------------------------------- #
#                                  Collate                                    #
# --------------------------------------------------------------------------- #
def collate_token_chunks(batch: List[np.ndarray]):
    """Pad variable-length token sequences and build corresponding labels.

    - Pad tokens with `eos_token_id`
    - Pad labels with -100 so they are ignored by the loss
    - Labels are not shifted here; the model handles shift internally
    """
    if not batch:
        raise ValueError("Empty batch in collate_token_chunks")

    max_len = max(int(arr.shape[0]) for arr in batch)
    pad_id = int(tokenizer.eos_token_id)

    def pad_toks(x: np.ndarray) -> np.ndarray:
        return np.pad(x, (0, max_len - x.shape[0]), constant_values=pad_id)

    def pad_labels(x: np.ndarray) -> np.ndarray:
        return np.pad(x, (0, max_len - x.shape[0]), constant_values=-100)

    toks = torch.tensor([pad_toks(x) for x in batch], dtype=torch.long)
    labels = torch.tensor([pad_labels(x) for x in batch], dtype=torch.long)
    return toks, labels


# --------------------------------------------------------------------------- #
#                                 Evaluate                                    #
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    total_sequences: Optional[int] = None,
) -> float:
    total_loss, total_tokens = 0.0, 0
    tmp_loss, tmp_tokens = 0.0, 0
    ct = 0
    total_batches = (
        math.ceil(total_sequences / loader.batch_size) if total_sequences else None
    )
    for toks, labels in tqdm.tqdm(loader, desc="Evaluating", total=total_batches):
        toks, labels = toks.to(device), labels.to(device)
        with autocast(dtype=torch.bfloat16):
            out = model(input_ids=toks, labels=labels)
        n_tokens = (labels != -100).sum().item()
        total_loss += out.loss.item() * n_tokens
        total_tokens += n_tokens
        tmp_loss += out.loss.item() * n_tokens
        tmp_tokens += n_tokens
        ct += 1
        if ct % 500 == 0:
            print(tmp_loss / tmp_tokens)
            tmp_loss, tmp_tokens = 0.0, 0
    return total_loss / total_tokens


# --------------------------------------------------------------------------- #
#                                    Main                                     #
# --------------------------------------------------------------------------- #

def main(args):
    # ----- model ------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_openlm_model_from_uuid(args.uuid)
    model.to(device, dtype=torch.bfloat16).eval()

    # ----- dataset ---------------------------------------------------------- #
    dataset = WikipediaTokenChunks(
        config=args.config,
        block_size=args.block_size,
        max_sequences=args.max_sequences,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_token_chunks,
    )

    # ----- evaluation ------------------------------------------------------- #
    loss = evaluate(
        model,
        loader,
        device,
        total_sequences=args.max_sequences,
    )
    print(f"Cross-entropy loss: {loss:.6f}\nPerplexity       : {math.exp(loss):.6f}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uuid",
        required=True,
        help="Datacomp-LM / Open-LM run UUID identifying the checkpoint to evaluate",
    )
    parser.add_argument(
        "--config",
        default="20231101.en",
        help="Hugging Face wikimedia/wikipedia configuration, e.g. 20231101.en",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Evaluation batch size (default: 8)"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=2048,
        help="Token chunk length for evaluation (default: 2048)",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Maximum number of token chunks to evaluate (default: None)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (IterableDataset; default: 0)",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=0,
        help="Streaming shuffle buffer size (0 disables; default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for shuffling (default: 0)",
    )
    main(parser.parse_args()) 