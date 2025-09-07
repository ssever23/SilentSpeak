import re
import sys
import copy
import numpy as np
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- BOOTSTRAP AV-HuBERT CLEANLY (avoid duplicate model registration) ---
import sys, importlib, types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAIRSEQ_DIR  = PROJECT_ROOT / "ext" / "av_hubert" / "fairseq"
USER_DIR     = PROJECT_ROOT / "ext" / "av_hubert" / "avhubert"

# Ensure vendored fairseq and avhubert are importable
sys.path.insert(0, str(FAIRSEQ_DIR))
sys.path.insert(0, str(USER_DIR))

# 1) Pre-import the top-level 'hubert' (this loads hubert.py once)
hubert_mod = importlib.import_module("hubert")

# 2) If hubert_pretraining is pulled in by hubert, capture it too
hubert_pre_mod = sys.modules.get("hubert_pretraining")

# 3) Alias them to the package-qualified names so relative imports reuse the same module
sys.modules.setdefault("avhubert.hubert", hubert_mod)
if hubert_pre_mod is not None:
    sys.modules.setdefault("avhubert.hubert_pretraining", hubert_pre_mod)

# Now it's safe to let fairseq load the user_dir package; it will reuse the aliases
from argparse import Namespace
from fairseq import utils as fs_utils
fs_utils.import_user_module(Namespace(user_dir=str(USER_DIR)))
# --- END BOOTSTRAP ---

from fairseq import checkpoint_utils, tasks



def decode_tokens(tokens, task, gen):
    """
    Handles SP ('▁'), GPT-2 ('Ġ'), subword-nmt ('@@'),
    letter/char labels (single-space between chars, multi-space between words),
    and '|' as the space symbol.
    """
    dictionary = task.target_dictionary
    ignore = set(getattr(gen, "symbols_to_strip_from_output", []))
    ignore.add(dictionary.pad())

    # 1) ids -> interim string of symbols (space-separated)
    s = dictionary.string(tokens.int().cpu(), extra_symbols_to_ignore=ignore)

    # 2) Heuristic detok by scheme
    if "▁" in s:
        # SentencePiece: remove separator spaces, turn '▁' into spaces
        s = s.replace(" ", "")
        s = s.replace("▁", " ").strip()
    elif "@@" " " in s or "@@" in s:
        # subword-nmt: remove continuation markers
        s = s.replace("@@ ", "").replace("@@", "")
        s = re.sub(r"\s{2,}", " ", s).strip()
    elif "Ġ" in s:
        # GPT-2/BPE: 'Ġ' marks a space before the token
        s = s.replace("Ġ", " ")
        s = re.sub(r"\s{2,}", " ", s).strip()
    elif "|" in s:
        # ltr-style vocab where '|' means space
        s = s.replace(" ", "")      # remove char separators
        s = s.replace("|", " ").strip()
    else:
        # Character/letter labels: single spaces inside words, multiple between words.
        # Remove single spaces between word chars, keep multi-spaces as word boundaries, then collapse.
        s = re.sub(r'(?<!\s)\s(?!\s)', '', s)  # kill lone intraword spaces
        s = re.sub(r'\s{2,}', ' ', s).strip()  # collapse multi-spaces to one

    # 3) Punctuation tidy-ups
    s = re.sub(r"\s*(['’`-])\s*", r"\1", s)        # that ' s -> that's ; co - op -> co-op
    s = re.sub(r"\s+([,.?!:;])", r"\1", s)         # remove space before punctuation
    s = re.sub(r"\s{2,}", " ", s).strip()
    
    return s


def predict_speech(model_path, npy_path, fps=25, win_sec=10.0, hop_sec=9.0,
                    beam=5, max_len_b=200, no_repeat_ngram_size=3):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path])
    model = models[0].eval().to(device)

    # Load npy -> [T, H, W] or [T, C, H, W]
    arr = np.load(npy_path, allow_pickle=True)
    if arr.ndim == 3:
        arr = arr[:, None, :, :]
    elif arr.ndim == 4 and arr.shape[1] != 1:
        arr = arr.mean(axis=1, keepdims=True)  # to 1-channel

    mean, std = 0.421, 0.165    # normalization parameters
    x = torch.from_numpy(arr).float()
    x = (x - mean) / (std + 1e-8)
    x = x.unsqueeze(0)                # [1, T, 1, H, W]
    x = x.permute(0, 2, 1, 3, 4)      # --> [1, 1, T, H, W]  (B, C, T, H, W)
    x = x.contiguous()
    x = x.to(device)

    T = x.shape[2]
    
    # Introspect limits
    try:
        print("Task max positions:", getattr(task, "max_positions", lambda: "unknown")())
    except Exception:
        pass
    print("Saved gen cfg:", saved_cfg.generation)

    # Build generator
    gen_args = copy.deepcopy(saved_cfg.generation)
    gen_args.beam = beam
    # Change some model configuration:
    if hasattr(gen_args, "max_len_b"): gen_args.max_len_b = max_len_b
    if hasattr(gen_args, "max_len_a"): gen_args.max_len_a = 0
    if hasattr(gen_args, "no_repeat_ngram_size"): gen_args.no_repeat_ngram_size = no_repeat_ngram_size
    gen = task.build_generator([model], gen_args)

    # Compute chunk indices
    win = int(round(win_sec * fps))
    hop = int(round(hop_sec * fps))
    if win <= 0 or hop <= 0:
        raise ValueError("win_sec and hop_sec must be > 0")

    pieces = []
    start = 0
    
    # Chunked inference
    while start < T:
        end = min(start + win, T)
        x_chunk = x[:, :, start:end, :, :]
        pad_mask = torch.zeros((1, end - start), dtype=torch.bool, device=device)

        sample = {
            "id": torch.tensor([0], device=device),
            "net_input": {
                "source": {"audio": None, 
                           "video": x_chunk},
                "padding_mask": pad_mask,
            },
        }

        with torch.no_grad():
            hypos = task.inference_step(gen, [model], sample)
            best = hypos[0][0]
            tokens = best.get("tokens", None)
            if tokens is not None:
                txt = decode_tokens(tokens, task, gen)
                pieces.append(txt)
            elif "words" in best:
                # Some checkpoints return words
                pieces.append(" ".join(best["words"]))
            else:
                pieces.append("")

        if end == T:
            break
        start += hop

    # Drop obvious duplicates at chunk seams
    text = " ".join(pieces)
    text = re.sub(r"\s+", " ", text).strip()
    return text


