#!/usr/bin/env python
"""
collect_grads_datacomp.py  (minimal diff from collect_grads.py)

Purpose
-------
Iterate over a *tokenised* WebDataset consisting of ``*.json.gz`` files that
contain a single field::

   {"tokens": [123, 456, …]}

and save per‑example gradients dL/dW for the model's ``lm_head.weight``.

Usage
-----
Exactly the same CLI as the original script – only the dataset contents differ:

    python collect_grads_datacomp.py \
        --wds-dir clustering/datacomp_tokshuf_shards \
        --ckpt  meta-llama/Llama-3-8B-Instruct \
        --out   clustering/datacomp_feats/grads.fp16 \
        --index clustering/datacomp_feats/index.tsv
"""
# ──────────────────────────────────────────────────────────────────────────
import argparse, json, uuid, gzip, io, os
from pathlib import Path
import numpy as np
import torch, tqdm, webdataset as wds
from transformers import AutoModelForCausalLM, AutoTokenizer
from lora.lora import LoRAHandler
from lora.logger import Logger
from torch.cuda.amp import GradScaler
from open_lm.utils.transformers.hf_model import OpenLMforCausalLM
from open_lm.main import load_model
# (rest of original imports remain unchanged)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -------------------- configuration -------------------- #
DTYPE_OUT  = np.float16
BATCH_SIZE = 16
# ------------------------------------------------------- #

# (all helper functions for loading HF / Open‑LM checkpoints remain unchanged)

# --- optional: only needed when you use --uuid -----------------------------
try:
    from open_lm.utils import download_cached          # S3/HTTP cache helper
    from open_lm.factory import build_open_lm          # model builder
except ImportError:
    download_cached = build_open_lm = None             # fallback for --ckpt

# ──────────────────────────── Data helpers ────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

def _load_tokens_from_sample(sample: dict) -> np.ndarray:
    """
    Accept a WebDataset sample whose payload is either
      * 'tokens.npy'  (old WebDataset)      or
      * '<anything>.json[.gz]' (tokshuf shards)
    and return it as a 1‑D NumPy array of dtype int32.
    """
    if "tokens.npy" in sample:                                     # legacy path
        return np.load(io.BytesIO(sample["tokens.npy"]))
    # otherwise assume JSON (optionally gzipped)
    key = next(k for k in sample if k.endswith("json") or k.endswith("json.gz"))
    raw = sample[key]
    if key.endswith(".gz"):
        raw = gzip.decompress(raw)
    obj = json.loads(raw)                                          # bytes → Python
    # Allow both {"tokens": [...]} and bare [...] formats.
    if isinstance(obj, list):
        tokens_list = obj
    elif isinstance(obj, dict):
        # Flexibly accept different possible keys
        if "tokens" in obj:
            tokens_list = obj["tokens"]
        elif "input_ids" in obj:
            tokens_list = obj["input_ids"]
        else:
            raise KeyError("JSON object missing 'tokens'/'input_ids' field")
    else:
        raise TypeError(f"Unsupported JSON payload type: {type(obj)}")

    return np.asarray(tokens_list, dtype=np.int32)

def load_hf_model(ckpt_path: str):
    """Standard HuggingFace loader (unchanged)."""
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    return model


def load_openlm_model_from_uuid(run_uuid: str):
    """Resolve a Datacomp-LM / Open-LM run UUID to a concrete checkpoint and
    return a ready-to-use torch.nn.Module.

    This re-implements what `eval/eval_openlm_ckpt.py` does without requiring
    any extra packages (in particular we avoid the deprecated
    `open_lm.factory.build_open_lm`).
    """
    from types import SimpleNamespace
    from open_lm.model import create_params
    from open_lm.utils.transformers.hf_config import OpenLMConfig

    # ------------------------------------------------------------------
    # 1) locate the metadata JSON describing the run
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parent.parent  # dclm_exp/
    exp_root = project_root / "exp_data" / "models"
    meta_path = next(exp_root.rglob(f"*{run_uuid}*.json"), None)
    if meta_path is None:
        raise FileNotFoundError(f"could not find metadata matching '*{run_uuid}*.json' in {exp_root}")

    meta = json.loads(meta_path.read_text())

    # ------------------------------------------------------------------
    # 2) Resolve paths for checkpoint and model-config
    # ------------------------------------------------------------------
    ckpt_path = Path(meta["checkpoint_url"]).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint file not found: {ckpt_path}")

    cfg_rel = meta["hyperparameters"].get("model")
    if cfg_rel is None:
        raise KeyError("Missing 'hyperparameters.model' entry in metadata JSON")
    cfg_path = (project_root / cfg_rel).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"model config file not found: {cfg_path}")

    # ------------------------------------------------------------------
    # 3) Build an Open-LM model ~ the same way the eval script does
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 4) Load checkpoint weights
    # ------------------------------------------------------------------
    load_args = SimpleNamespace(resume=str(ckpt_path), fsdp=False, distributed=False, seed=0)
    load_model(load_args, wrapper.model, different_seed=True)

    return wrapper


def get_model(args):
    """Choose the proper load routine based on the flags."""
    if args.uuid is not None:
        return load_openlm_model_from_uuid(args.uuid)
    if args.ckpt is not None:
        return load_hf_model(args.ckpt)
    raise ValueError("must supply either --uuid or --ckpt")

def prepare_model(model, args):
    model_parameters = sum(p.numel() for p in model.parameters())
    handler = LoRAHandler(rank=args.lora_rank)
    # init logger with model parameters and number of blocks
    logger = Logger(model_parameters, args.num_blocks)
    # LoRAHandler expects the bare Transformer (with attribute n_layers).
    core_model = model.model if hasattr(model, "model") else model
    handler.add_lora(core_model, logger, BATCH_SIZE)
    logger.init_grads(BATCH_SIZE, args.lora_rank)

    print(
        f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    for name, param in model.named_parameters():
        if "logix_lora_B" not in name:
            param.requires_grad = False
    print(
        f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device,dtype=torch.bfloat16)
    model.eval()
    #torch.use_deterministic_algorithms(True)
    return model, handler, logger

def collate(batch):
    """
    Pad‑and‑stack a list of WebDataset samples into tensors suitable for a
    HuggingFace causal‑LM forward pass (inputs + labels).
    - Inputs are left‑padded with EOS so every sequence has equal length.
    - Labels are **identical** to the padded inputs (HF does the 1‑token shift
      internally). Padding positions are set to -100 so they do not contribute
      to the loss.
    """
    toks = [ _load_tokens_from_sample(b) for b in batch ]
    L    = max(len(t) for t in toks)

    pad_id     = tokenizer.eos_token_id
    pad_fn     = lambda arr: np.pad(arr, (0, L-len(arr)), constant_values=pad_id)
    toks_padded = torch.tensor([ pad_fn(t) for t in toks ], dtype=torch.long)

    labels = toks_padded.clone()
    labels[ labels == pad_id ] = -100                               # ignore pad

    keys = [ b["__key__"] for b in batch ]
    return toks_padded, labels, keys

# ─────────────────────────────── Main ─────────────────────────────────────
def main(args):
    # ---------- model ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine a process-unique device index even when CUDA_VISIBLE_DEVICES
    # remaps GPUs to 0..N inside the process. Prefer distributed env vars,
    # then fall back to the first entry of CUDA_VISIBLE_DEVICES, and finally
    # to torch.cuda.current_device().
    import os
    if torch.cuda.is_available():
        env_rank = os.environ.get("LOCAL_RANK") or os.environ.get("RANK")
        if env_rank is not None:
            device_num = int(env_rank)
        else:
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cvd is not None and cvd != "":
                device_num = int(cvd.split(",")[0])
            else:
                device_num = torch.cuda.current_device()
    else:
        device_num = 0

    model, handler, logger = prepare_model(get_model(args), args)

    # ---------- dataset ----------
    wds_dir_path = Path(args.wds_dir)
    if not wds_dir_path.is_absolute() and not wds_dir_path.exists():
        alt_path = Path(__file__).resolve().parent / wds_dir_path
        if alt_path.exists():
            wds_dir_path = alt_path
        else:
            raise FileNotFoundError(f"dataset directory not found: {args.wds_dir}")

    tar_files = sorted(wds_dir_path.glob("*.tar"))
    NUM_SHARDS = 100
    tar_files = tar_files[device_num * NUM_SHARDS: (device_num + 1) * NUM_SHARDS]
    if not tar_files:
        raise FileNotFoundError(f"No .tar shards found in {wds_dir_path}")

    # A decode() stage is not strictly necessary because we handle gzip/json
    # manually in _load_tokens_from_sample(), but it doesn't hurt:
    ds     = wds.WebDataset([str(p) for p in tar_files])
    loader = wds.WebLoader(ds, batch_size=BATCH_SIZE, num_workers=64,
                           collate_fn=collate)

    # Derive an (approximate) sample count from the manifest, exactly as before.
    '''manifest_path = wds_dir_path / "manifest.jsonl"
    with open(manifest_path) as mf:
        N_est = int(mf.read().split(",")[1])'''
    N_est = NUM_SHARDS * 8192

    grads = np.memmap(args.out + f"_part_{device_num}", mode="w+", dtype=DTYPE_OUT,
                      shape=(N_est, logger.grads.shape[0], logger.grads.shape[2]))
    index = open(args.index + f"_part_{device_num}", "w")
    row_offset = 0
    scaler = GradScaler()
    device = next(model.parameters()).device
    # get device number for saving grads

    for step, (toks, labels, keys) in enumerate(
            tqdm.tqdm(loader, total=N_est // BATCH_SIZE)):
        toks, labels = toks.to(device), labels.to(device)
        with torch.enable_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=toks, labels=labels)
                out.loss.mul_(1024)
                out.loss.backward()
            features = (logger.grads.detach().cpu()
                        .to(torch.float32).numpy().transpose(1, 0, 2))
        logger.grads.zero_()
        grads[row_offset:row_offset + features.shape[0]] = features
        for k in keys:
            index.write(f"{row_offset}\t{k}\n")
            row_offset += 1

    grads.flush()
    index.close()

# ------------------------------ CLI ---------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wds-dir", required=True,
                    help="directory containing tokenised WebDataset shards")
    ap.add_argument("--uuid",  help="Datacomp-LM run UUID (overrides --ckpt)")
    ap.add_argument("--ckpt",  help="HuggingFace checkpoint path or Hub ID")
    ap.add_argument("--out",   default="clustering/datacomp_feats/grads.fp16",
                    help="output memmap filename")
    ap.add_argument("--index", default="clustering/datacomp_feats/index.tsv",
                    help="tsv mapping row_id → sample key")
    ap.add_argument("--lora-rank",  type=int, default=128)
    ap.add_argument("--num-blocks", type=int, default=8)
    args = ap.parse_args()

    if (args.uuid is None) == (args.ckpt is None):
        raise SystemExit("exactly one of --uuid or --ckpt is required")
    main(args)
