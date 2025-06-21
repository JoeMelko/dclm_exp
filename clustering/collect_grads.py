#!/usr/bin/env python
"""
collect_grads.py
----------------
Iterate over a *tokenised* WebDataset (tokens.npy / labels.npy) and save
per-example gradients dL/dW for the model's ``lm_head.weight`` at a fixed 400 M
checkpoint.

Two loading modes
-----------------
* **--uuid**  : give the run UUID produced by Datacomp-LM/Open-LM training;
                the script resolves the JSON in ``exp_data/models`` and loads
                the exact checkpoint the evaluation code would.
* **--ckpt**  : (legacy) path or HF Hub ID pointing to a standard HuggingFace
                directory with ``pytorch_model*.bin`` etc.

Outputs
-------
grads.fp16        mem-map  (N_samples, vocab * hidden)  ~ float16
sample_index.tsv  row_id <TAB> sample_key   (key == tokens.npy basename)
"""
import argparse, json, uuid
from pathlib import Path
import os
# Ensure deterministic CuBLAS reproducibility per PyTorch docs
'''if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"'''
# disable fused kernels
os.environ["XFORMERS_DISABLE_SWIGLU"] = "1"
import numpy as np
import torch, tqdm, webdataset as wds
from transformers import AutoModelForCausalLM
from lora.lora import LoRAHandler
from lora.logger import Logger
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer
from open_lm.utils.transformers.hf_model import OpenLMforCausalLM
from open_lm.main import load_model
import io

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- optional: only needed when you use --uuid -----------------------------
try:
    from open_lm.utils import download_cached          # S3/HTTP cache helper
    from open_lm.factory import build_open_lm          # model builder
except ImportError:
    download_cached = build_open_lm = None             # fallback for --ckpt
# ---------------------------------------------------------------------------

# -------------------- configuration -------------------- #
DTYPE_OUT     = np.float16
BATCH_SIZE    = 16                        # fits in 80 GB on an H100
# ------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#                                Loaders                                      #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
#                              Data helpers                                   #
# --------------------------------------------------------------------------- #
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
def collate(batch):
    toks   = [np.load(io.BytesIO(b["tokens.npy"])) for b in batch]
    labels = [np.load(io.BytesIO(b["labels.npy"])) for b in batch]
    L      = max(len(t) for t in toks)
    pad_toks    = lambda x: np.pad(x, (0, L - len(x)), constant_values = tokenizer.eos_token_id)
    pad_labels  = lambda x: np.pad(x, (0, L - len(x)), constant_values = -100)
    toks   = torch.tensor([pad_toks(t) for t in toks],   dtype = torch.long)
    labels = torch.tensor([pad_labels(l) for l in labels], dtype = torch.long)
    return toks, labels, [b["__key__"] for b in batch]


# --------------------------------------------------------------------------- #
#                                   Main                                      #
# --------------------------------------------------------------------------- #
def main(args):
    # ---------- model ----------
    model, handler, logger = prepare_model(get_model(args), args)
    
    # ---------- dataset ----------
    # Resolve dataset dir; if not absolute, check relative to clustering/
    wds_dir_path = Path(args.wds_dir)
    if not wds_dir_path.is_absolute() and not wds_dir_path.exists():
        alt_path = Path(__file__).resolve().parent / wds_dir_path
        if alt_path.exists():
            wds_dir_path = alt_path
        else:
            raise FileNotFoundError(f"Could not find dataset directory '{args.wds_dir}' or '{alt_path}'.")

    tar_files = sorted(wds_dir_path.glob("*.tar"))
    if len(tar_files) == 0:
        raise FileNotFoundError(f"No .tar shards found in {wds_dir_path}")

    ds = wds.WebDataset([str(p) for p in tar_files])
    loader = wds.WebLoader(ds, batch_size=BATCH_SIZE, num_workers=64, collate_fn=collate)

    manifest_path = wds_dir_path / "manifest.jsonl"
    with open(manifest_path) as mf:
        N_est = int(mf.read().split(",")[1])

    grads = np.memmap(args.out, mode = "w+", dtype = DTYPE_OUT,
                      shape = (N_est, logger.grads.shape[0], logger.grads.shape[2]))
    index = open(args.index, "w")
    row_offset   = 0
    scaler = GradScaler()

    device = next(model.parameters()).device

    for step, (toks, labels, keys) in enumerate(tqdm.tqdm(loader, total = N_est // BATCH_SIZE)):
        toks = toks.to(device)
        labels = labels.to(device)
        with torch.enable_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out  = model(input_ids = toks, labels = labels)
                breakpoint()
                print(out.loss)
                out.loss.mul_(1024)
                out.loss.backward()
            features = (
                logger.grads.detach().cpu().to(torch.float32).numpy().transpose(1, 0, 2)
            )
        logger.grads.zero_()
        grads[row_offset:row_offset + features.shape[0]] = features
        # write index keys
        for k in keys:
            index.write(f"{row_offset}\t{k}\n")
            row_offset += 1

    grads.flush()
    index.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wds-dir", required = True,
                    help = "directory containing tokenised WebDataset shards")
    ap.add_argument("--uuid",
                    help = "Datacomp-LM run UUID (overrides --ckpt)")
    ap.add_argument("--ckpt",
                    help = "HuggingFace checkpoint path or Hub ID")
    ap.add_argument("--out",   default = "clustering/openhermes_feats/grads.fp16_mul_1024",
                    help = "output memmap filename")
    ap.add_argument("--index", default = "clustering/openhermes_feats/index.tsv",
                    help = "tsv mapping row_id → sample key")
    ap.add_argument("--lora-rank", type=int, default=128,
                    help = "LORA rank")
    ap.add_argument("--num-blocks", type=int, default=8,
                    help = "number of blocks")
    args = ap.parse_args()

    # sanity: enforce mutually exclusive args
    if (args.uuid is None) == (args.ckpt is None):
        raise SystemExit("exactly one of --uuid or --ckpt is required")

    main(args)
