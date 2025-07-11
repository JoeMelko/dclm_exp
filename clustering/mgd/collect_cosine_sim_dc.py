#!/usr/bin/env python
"""
collect_cosine_sim_dc.py

Purpose
-------
Iterate over a *tokenised* WebDataset consisting of ``*.json.gz`` samples that
contain ``{"tokens": [...]}`` and, for every batch
(1) compute per-example gradient features via LoRA's `Logger`,
(2) average the features across the batch,
(3) compute the cosine similarity against a **target vector** provided on the
    command line, and
(4) store all similarities in a ``.npz`` file named
    ``{<wds_dir-basename>}.npz`` inside a sub-directory
    ``iter_<iter>`` that is created under the output directory.

Only minimal changes compared to ``collect_grads_dc.py``:
* new CLI flag ``--target-vector`` – path to ``.npy`` containing a 1-D target
  vector of shape ``(num_blocks * lora_rank,)`` **that has already been
  whitened and L2-normalised**.
* new CLI flag ``--iter`` with default ``0`` – controls output file name
* compared to ``collect_grads_dc.py`` we now collect per-sample cosine
  similarities rather than raw gradients.
* NOTE: The previous ``--whiteners`` argument has been removed – the target
  vector is assumed to be in the same feature space as the gradients, so
  no additional whitening is performed at runtime.
"""
# ──────────────────────────────────────────────────────────────────────────
import argparse, json, gzip, io, sys
from pathlib import Path

# ---------------------------------------------------------------------
# Ensure parent directory (clustering/) is on PYTHONPATH so that
# `import lora.*` works when this script is executed from clustering/mgd
# ---------------------------------------------------------------------
PARENT_DIR = Path(__file__).resolve().parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import numpy as np
import torch, tqdm, webdataset as wds
from transformers import AutoModelForCausalLM, AutoTokenizer
from lora.lora import LoRAHandler
from lora.logger import Logger
from open_lm.utils.transformers.hf_model import OpenLMforCausalLM
from open_lm.main import load_model

# allow TF32 for speed
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -------------------- configuration -------------------- #
BATCH_SIZE = 8
# ------------------------------------------------------- #

# --- optional: only needed when you use --uuid -----------------------------
try:
    from open_lm.utils import download_cached          # S3/HTTP cache helper
    from open_lm.factory import build_open_lm          # model builder
except ImportError:                                    # fallback for --ckpt
    download_cached = build_open_lm = None

# ──────────────────────────── Data helpers ────────────────────────────────
# We only need the tokenizer for padding-id lookup.
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

def _load_tokens_from_sample(sample: dict) -> np.ndarray:
    """Extract the token list from a WebDataset sample and return it as
    ``np.ndarray`` of dtype ``int32``."""
    if "tokens.npy" in sample:  # legacy path
        return np.load(io.BytesIO(sample["tokens.npy"]))

    key = next(k for k in sample if k.endswith("json") or k.endswith("json.gz"))
    raw = sample[key]
    if key.endswith(".gz"):
        raw = gzip.decompress(raw)
    obj = json.loads(raw)

    if isinstance(obj, list):
        tokens_list = obj
    elif isinstance(obj, dict):
        if "tokens" in obj:
            tokens_list = obj["tokens"]
        elif "input_ids" in obj:
            tokens_list = obj["input_ids"]
        else:
            raise KeyError("JSON object missing 'tokens'/'input_ids' field")
    else:
        raise TypeError(f"Unsupported JSON payload type: {type(obj)}")

    return np.asarray(tokens_list, dtype=np.int32)

# ──────────────────────────── Model loading ───────────────────────────────

def load_hf_model(ckpt_path: str):
    """Load a HuggingFace causal-LM checkpoint with Flash-Attention enabled."""
    return AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )


def load_openlm_model_from_uuid(run_uuid: str):
    """Resolve an Open-LM run UUID to a checkpoint and build the model."""
    from types import SimpleNamespace
    from open_lm.model import create_params
    from open_lm.utils.transformers.hf_config import OpenLMConfig

    # Resolve project root dynamically: ascend the directory tree until
    # we find an `exp_data` directory.  This works whether the script
    # lives under `clustering/mgd/` or is moved elsewhere.
    _here = Path(__file__).resolve()
    project_root = next((p for p in _here.parents if (p / "exp_data").exists()), _here.parent.parent.parent)
    exp_root = project_root / "exp_data" / "models"
    meta_path = next(exp_root.rglob(f"*{run_uuid}*.json"), None)
    if meta_path is None:
        raise FileNotFoundError(
            f"could not find metadata matching '*{run_uuid}*.json' in {exp_root}"
        )

    meta = json.loads(meta_path.read_text())
    ckpt_path = Path(meta["checkpoint_url"]).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint file not found: {ckpt_path}")

    cfg_rel = meta["hyperparameters"].get("model")
    if cfg_rel is None:
        raise KeyError("Missing 'hyperparameters.model' entry in metadata JSON")
    cfg_path = (project_root / cfg_rel).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"model config file not found: {cfg_path}")

    dummy_args = SimpleNamespace(
        model=str(cfg_path),
        model_norm=meta["hyperparameters"].get("norm", "gain_only_lp_layer_norm"),
        attn_name="torch_attn",
        attn_activation=None,
        attn_seq_scalar=None,
        attn_seq_scalar_alpha=None,
        qk_norm=meta["hyperparameters"].get("qk_norm", False),
        positional_embedding_type="rotary",
        ffn_type="swiglu_torch",
        moe_num_experts=meta["hyperparameters"].get("moe_num_experts", 8),
        moe_loss_weight=0.1,
        moe_expert_model_parallelism=False,
        moe_weight_parallelism=False,
        moe_capacity_factor=1.25,
        moe_freq=0,
        moe_top_k=2,
    )

    params = create_params(dummy_args)
    wrapper = OpenLMforCausalLM(OpenLMConfig(params))

    load_args = SimpleNamespace(resume=str(ckpt_path), fsdp=False, distributed=False, seed=0)
    load_model(load_args, wrapper.model, different_seed=True)
    return wrapper


def get_model(args):
    if args.uuid is not None:
        return load_openlm_model_from_uuid(args.uuid)
    if args.ckpt is not None:
        return load_hf_model(args.ckpt)
    raise ValueError("must supply either --uuid or --ckpt")


# ──────────────────────────── LoRA helpers ────────────────────────────────

def prepare_model(model, args):
    model_parameters = sum(p.numel() for p in model.parameters())
    handler = LoRAHandler(rank=args.lora_rank)
    logger = Logger(model_parameters, args.num_blocks)
    core_model = model.model if hasattr(model, "model") else model
    handler.add_lora(core_model, logger, BATCH_SIZE)
    logger.init_grads(BATCH_SIZE, args.lora_rank)

    # freeze all except LoRA params
    for name, param in model.named_parameters():
        if "logix_lora_B" not in name:
            param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)
    model.eval()
    return model, handler, logger

# ───────────────────────────── Data collation ─────────────────────────────

def collate(batch):
    toks = [_load_tokens_from_sample(b) for b in batch]
    L = max(len(t) for t in toks)

    pad_id = tokenizer.eos_token_id
    pad_fn = lambda arr: np.pad(arr, (0, L - len(arr)), constant_values=pad_id)
    toks_padded = torch.tensor([pad_fn(t) for t in toks], dtype=torch.long)

    labels = toks_padded.clone()
    labels[labels == pad_id] = -100  # ignore pad

    return toks_padded, labels

# ─────────────────────────────── Main ─────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, handler, logger = prepare_model(get_model(args), args)

    # -------------------------------------------------------------------
    # Load target vector and whitener matrices **onto the GPU** so that
    # all subsequent computations stay on device until the very end.
    # -------------------------------------------------------------------
    device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_vec = (
        torch.from_numpy(np.load(args.target_vector).astype(np.float32))
        .to(device_t)
        .view(-1)
    )  # already whitened + L2-normalised

    # Ensure the target vector length matches the expected feature dimension
    # given by ``num_blocks * lora_rank`` supplied on the command line. No
    # runtime whitening is required because both the gradients and the target
    # vector are assumed to be **already** whitened and live in the same
    # feature space.

    expected_dim = 2 * args.num_blocks * args.lora_rank * args.lora_rank
    if target_vec.numel() != expected_dim:
        raise ValueError(
            f"Target vector length {target_vec.numel()} does not equal the "
            f"expected dimension {args.num_blocks} * {args.lora_rank} = {expected_dim}."
        )

    # dataset
    wds_dir_path = Path(args.wds_dir)
    if not wds_dir_path.is_absolute() and not wds_dir_path.exists():
        alt_path = Path(__file__).resolve().parent / wds_dir_path
        if alt_path.exists():
            wds_dir_path = alt_path
        else:
            raise FileNotFoundError(f"dataset directory not found: {args.wds_dir}")

    tar_files = sorted(wds_dir_path.glob("*.tar"))
    if not tar_files:
        raise FileNotFoundError(f"No .tar shards found in {wds_dir_path}")

    ds = wds.WebDataset([str(p) for p in tar_files])
    loader = wds.WebLoader(ds, batch_size=BATCH_SIZE, num_workers=8, collate_fn=collate)

    dot_products_gpu = []  # accumulate torch scalars on GPU
    l2_norms_gpu = []  # accumulate torch scalars on GPU
    device_t = next(model.parameters()).device

    for toks, labels in tqdm.tqdm(loader, desc="batches"):
        # ------------------------------------------------------------------
        # If the current batch is smaller than the requested BATCH_SIZE we
        # consider the dataset exhausted and stop. Processing an incomplete
        # batch would break shape assumptions further down (e.g., the LoRA
        # logger collects gradients for exactly `BATCH_SIZE` examples).
        # ------------------------------------------------------------------
        if toks.shape[0] < BATCH_SIZE:
            print(f"Encountered incomplete batch of size {toks.shape[0]} < {BATCH_SIZE}; stopping featurization loop.")
            break

        toks, labels = toks.to(device_t), labels.to(device_t)
        with torch.enable_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=toks, labels=labels)
                out.loss.mul_(1024)
                out.loss.backward()
            # collect grads → (num_blocks, batch, rank)
            # Clone to decouple from the buffer; otherwise zero_() would
            # invalidate the cached values.
            features = logger.grads.detach().clone().to(torch.float32)  # on GPU

        # Reset gradient buffer for next step **before any further GPU ops**
        logger.grads.zero_()

        # ------------------------------------------------------------------
        # Per-sample processing (no batch average)
        # 1. Rearrange to (batch, num_blocks, rank)
        # 2. Flatten gradient features  → (batch, num_blocks * rank)
        # 3. Dot product with target vector, and its L2-norm.
        # ------------------------------------------------------------------

        feats_per_sample = features.permute(1, 0, 2)  # (batch, B, d)

        # Basic sanity-check: gradient tensor should have the configured
        # num_blocks × lora_rank shape.
        if feats_per_sample.shape[1] != 2 * args.num_blocks or feats_per_sample.shape[2] != args.lora_rank * args.lora_rank:
            raise ValueError(
                f"Unexpected feature shape {tuple(feats_per_sample.shape)} – "
                f"expected (*, {2 * args.num_blocks}, {args.lora_rank * args.lora_rank})."
            )

        # Flatten gradient features  → (batch, num_blocks * rank)
        flat_feats = feats_per_sample.reshape(feats_per_sample.size(0), -1)

        if flat_feats.shape[1] != target_vec.numel():
            raise ValueError(
                f"Target vector length {target_vec.numel()} != gradient feature length {flat_feats.shape[1]}"
            )

        # Dot product (non-normalised) with target vector
        dots = torch.matmul(flat_feats, target_vec)  # (batch,)

        # L2 norm of each flattened grad
        norms = torch.linalg.norm(flat_feats, dim=1)  # (batch,)

        dot_products_gpu.append(dots)
        l2_norms_gpu.append(norms)
        if len(l2_norms_gpu) > args.max_items:
            # dont waste compute on massive clusters
            break

    # Concatenate results, move to CPU once, convert to NumPy
    dot_products = torch.cat(dot_products_gpu).cpu().numpy().astype(np.float32)
    l2_norms = torch.cat(l2_norms_gpu).cpu().numpy().astype(np.float32)

    # Resolve and create output directory if it does not exist
    # Create sub-directory "iter_<iter>" under the base output directory.
    out_dir_path = Path(args.out_dir).expanduser() / f"iter_{args.iter}"
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Output file no longer includes the iteration index in its name.
    out_path = out_dir_path / f"{wds_dir_path.name}.npz"
    np.savez(out_path, dot=dot_products, l2=l2_norms)
    print(
        f"saved {dot_products.shape[0]} per-sample dot products and norms → {out_path}"
    )


# ------------------------------ CLI ---------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wds-dir", required=True, help="directory with tokenised WebDataset shards")
    ap.add_argument("--target-vector", required=True, help="path to .npy containing flattened target vector (already whitened + L2-normalised)")
    ap.add_argument("--uuid", help="Datacomp-LM run UUID (overrides --ckpt)")
    ap.add_argument("--ckpt", help="HuggingFace checkpoint path or Hub ID")
    ap.add_argument("--iter", type=int, default=0, help="iteration index used in output file name")
    ap.add_argument("--lora-rank", type=int, default=128)
    ap.add_argument("--num-blocks", type=int, default=8)
    ap.add_argument("--out-dir", default="clustering/mgd", help="Directory where output will be written (default: clustering/mgd/)")
    ap.add_argument("--max-items", type=int, default=500, help="maximum number of batches processed before stopping early (default: 500)")
    args = ap.parse_args()

    if (args.uuid is None) == (args.ckpt is None):
        raise SystemExit("exactly one of --uuid or --ckpt is required")

    main(args) 