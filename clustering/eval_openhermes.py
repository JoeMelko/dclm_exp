#!/usr/bin/env python
"""
eval_openhermes.py
------------------
Compute the token-level cross-entropy loss (and perplexity) of an Open-LM
checkpoint – identified by its Datacomp-LM run UUID – on the tokenised
OpenHermes WebDataset.

Supports efficient approximate evaluation by loading a larger full batch
(`--full-batch-size`) and randomly subsampling `--batch-size` examples within
that batch to run the model forward pass. The remaining examples are discarded.
This reduces compute while preserving an unbiased estimate over the sampled
subset.
"""
import argparse, json, math, io, random
from pathlib import Path

import numpy as np
import torch, tqdm, webdataset as wds
from torch.cuda.amp import autocast
from transformers import AutoTokenizer

# Enable TF-32 when available – safe speedup, no impact on results
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# --------------------------------------------------------------------------- #
#                         Model loader (UUID only)                            #
# --------------------------------------------------------------------------- #
def load_openlm_model_from_uuid(run_uuid: str) -> torch.nn.Module:
    """
    Resolve a Datacomp-LM / Open-LM run UUID to a checkpoint and return the
    corresponding `torch.nn.Module` ready for inference.
    """
    from types import SimpleNamespace

    from open_lm.model import create_params
    from open_lm.main import load_model
    from open_lm.utils.transformers.hf_config import OpenLMConfig
    from open_lm.utils.transformers.hf_model import OpenLMforCausalLM

    project_root = Path(__file__).resolve().parent.parent        # dclm_exp/
    exp_root     = project_root / "exp_data" / "models"

    meta_path = next(exp_root.rglob(f"*{run_uuid}*.json"), None)
    if meta_path is None:
        raise FileNotFoundError(
            f"Could not find metadata matching '*{run_uuid}*.json' in {exp_root}"
        )
    meta = json.loads(meta_path.read_text())

    ckpt_path = Path(meta["checkpoint_url"]).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    cfg_rel  = meta["hyperparameters"].get("model")
    if cfg_rel is None:
        raise KeyError("Missing 'hyperparameters.model' entry in metadata JSON")
    cfg_path = (project_root / cfg_rel).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Model config file not found: {cfg_path}")

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
    model  = OpenLMforCausalLM(OpenLMConfig(params))

    load_args = SimpleNamespace(resume=str(ckpt_path),
                                fsdp=False, distributed=False, seed=0)
    load_model(load_args, model.model, different_seed=True)

    return model


# --------------------------------------------------------------------------- #
#                         Data loading helpers                                #
# --------------------------------------------------------------------------- #
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

def collate(batch):
    """
    WebDataset collate fn – pads tokens/labels to equal length.
    Labels are padded with -100 so they are ignored by the loss.
    """
    toks   = [np.load(io.BytesIO(b["tokens.npy"])) for b in batch]
    labels = [np.load(io.BytesIO(b["labels.npy"])) for b in batch]
    L = max(len(t) for t in toks)

    pad_toks   = lambda x: np.pad(x, (0, L - len(x)),
                                  constant_values=tokenizer.eos_token_id)
    pad_labels = lambda x: np.pad(x, (0, L - len(x)), constant_values=-100)

    toks   = torch.tensor([pad_toks(t)   for t in toks],   dtype=torch.long)
    labels = torch.tensor([pad_labels(l) for l in labels], dtype=torch.long)
    return toks, labels


# --------------------------------------------------------------------------- #
#                               Evaluation                                    #
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def evaluate(model: torch.nn.Module,
             loader: torch.utils.data.DataLoader,
             device: torch.device,
             sample_batch_size: int) -> float:
    total_loss, total_tokens = 0.0, 0
    tmp_loss, tmp_tokens = 0.0, 0
    ct = 0
    for toks, labels in tqdm.tqdm(loader, desc="Evaluating"):
        # Subsample a random subset of examples from the full batch
        full_bs = toks.size(0)
        use_bs = min(sample_batch_size, full_bs)
        if use_bs < full_bs:
            perm = torch.randperm(full_bs, device=toks.device)
            idx = perm[:use_bs]
            toks_sub = toks.index_select(0, idx)
            labels_sub = labels.index_select(0, idx)
        else:
            toks_sub, labels_sub = toks, labels

        toks_sub, labels_sub = toks_sub.to(device), labels_sub.to(device)
        with autocast(dtype=torch.bfloat16):
            out = model(input_ids=toks_sub, labels=labels_sub)
        # `out.loss` is mean over non-ignored labels within the sub-batch
        n_tokens = (labels_sub != -100).sum().item()
        total_loss   += out.loss.item() * n_tokens
        total_tokens += n_tokens
        tmp_loss += out.loss.item() * n_tokens
        tmp_tokens += n_tokens
        ct += 1
        if ct % 500 == 0 and tmp_tokens > 0:
            print(tmp_loss / tmp_tokens)
            tmp_loss, tmp_tokens = 0.0, 0
    if total_tokens == 0:
        return float("nan")
    return total_loss / total_tokens


def main(args):
    # ----- model ---------------------------------------------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ----- seeding -------------------------------------------------------- #
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model  = load_openlm_model_from_uuid(args.uuid)
    model.to(device, dtype=torch.bfloat16).eval()

    # ----- dataset -------------------------------------------------------- #
    wds_dir = Path(args.wds_dir)
    if not wds_dir.is_absolute() and not wds_dir.exists():
        alt = Path(__file__).resolve().parent / wds_dir
        wds_dir = alt if alt.exists() else wds_dir
    if not wds_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {wds_dir}")

    shards = sorted(wds_dir.glob("*.tar"))
    if not shards:
        raise FileNotFoundError(f"No '.tar' shards found in {wds_dir}")

    ds = wds.WebDataset([str(p) for p in shards])
    loader = wds.WebLoader(ds,
                           batch_size=args.full_batch_size,
                           num_workers=8,
                           collate_fn=collate)

    # ----- evaluation ----------------------------------------------------- #
    loss = evaluate(model, loader, device, sample_batch_size=args.batch_size)
    print(f"Cross-entropy loss: {loss:.6f}\nPerplexity       : {math.exp(loss):.6f}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wds-dir", required=True,
                        help="Directory containing tokenised OpenHermes shards (*.tar)")
    parser.add_argument("--uuid", required=True,
                        help="Datacomp-LM / Open-LM run UUID identifying the checkpoint to evaluate")
    parser.add_argument("--seed", type=int, default=43,
                        help="Random seed for reproducible subsampling (default: 43)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Number of examples to subsample per full batch for the forward pass (default: 16)")
    parser.add_argument("--full-batch-size", type=int, default=128,
                        help="Full batch size to load/collate before subsampling (default: 64)")
    main(parser.parse_args())
