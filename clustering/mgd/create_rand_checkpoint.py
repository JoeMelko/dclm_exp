#!/usr/bin/env python
"""
create_rand_checkpoint.py
-------------------------
Utility script to generate a *randomly initialised* checkpoint that is compatible
with Open-LM / Datacomp-LM evaluation scripts expecting a metadata JSON.

Given a metadata JSON file (the one normally stored in `exp_data/models/…`), the
script will:

1. Read the JSON to obtain the YAML config and hyper-parameters
2. Build the corresponding model, applying a deterministic seed
3. Save the fresh weights to a new checkpoint file (`.pt`)
4. Write a *copy* of the metadata JSON whose `checkpoint_url` field now points
   to the newly created checkpoint

You can then pass this **new** JSON to existing pipelines (e.g. `get_target.py`
with `--uuid <new_json_path>`) and obtain deterministic random initialisation
without modifying their code.

Example:
    python create_rand_checkpoint.py \
        --meta exp_data/models/run_123/meta.json \
        --seed 42 \
        --out-dir exp_data/models/run_123_rand
"""
import argparse, json, random, shutil, sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure parent directory (clustering/) is on PYTHONPATH so `lora`, `open_lm`,
# etc. are resolvable no matter where we run this script from.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve()
CLUSTERING_DIR = HERE.parent
if str(CLUSTERING_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTERING_DIR))

from open_lm.model import create_params
from open_lm.utils.transformers.hf_model import OpenLMforCausalLM
from open_lm.utils.transformers.hf_config import OpenLMConfig

# ---------------------------------------------------------------------------
#                      Deterministic seeding helper
# ---------------------------------------------------------------------------

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
#                              Main routine
# ---------------------------------------------------------------------------

def build_model_from_meta(meta_path: Path):
    """Return a *random-initialised* `OpenLMforCausalLM` using the config in `meta_path`."""
    meta = json.loads(meta_path.read_text())

    cfg_rel = meta["hyperparameters"].get("model")
    if cfg_rel is None:
        raise KeyError("Missing 'hyperparameters.model' entry in metadata JSON")

    cfg_path = (
        (meta_path.parent.parent.parent / cfg_rel).resolve()
        if not Path(cfg_rel).is_absolute()
        else Path(cfg_rel)
    )
    if not cfg_path.exists():
        raise FileNotFoundError(f"Model config YAML not found: {cfg_path}")

    args = SimpleNamespace(
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

    return OpenLMforCausalLM(OpenLMConfig(create_params(args)))


def main(args):
    meta_path = Path(args.meta).expanduser().resolve()
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata JSON not found: {meta_path}")

    # ------------------------------------------------------------------
    # Seeding before model construction ensures determinism
    # ------------------------------------------------------------------
    set_global_seed(args.seed)

    model = build_model_from_meta(meta_path)
    # ensure we save the *inner* transformer (so parameter names do not start with 'model.')
    core_model = model.model if hasattr(model, "model") else model

    # ------------------------------------------------------------------
    # Prepare output paths
    # ------------------------------------------------------------------
    out_dir = Path(args.out_dir) if args.out_dir else meta_path.parent / f"randinit_seed{args.seed}"
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "model_randinit.pt"
    torch.save(core_model.state_dict(), ckpt_path)

    # ------------------------------------------------------------------
    # Write new meta JSON pointing to the fresh checkpoint
    # ------------------------------------------------------------------
    new_meta = json.loads(meta_path.read_text())  # deep copy via serialisation
    new_meta["checkpoint_url"] = str(ckpt_path)

    new_meta_path = out_dir / meta_path.name
    new_meta_path.write_text(json.dumps(new_meta, indent=2))

    print("\nRandom-initialised checkpoint created:\n  weights :", ckpt_path,
          "\n  meta    :", new_meta_path,
          "\n\nPass the new JSON to your scripts, e.g.:\n  --uuid", new_meta_path, sep="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a random-init Open-LM checkpoint compatible with existing JSON metadata.")
    parser.add_argument("--meta", required=True, help="Path to the original metadata JSON file")
    parser.add_argument("--seed", type=int, default=0, help="Global RNG seed (default: 0)")
    parser.add_argument("--out-dir", help="Directory where checkpoint & new JSON will be saved (defaults to <meta_dir>/randinit_seed<seed>)")

    main(parser.parse_args()) 