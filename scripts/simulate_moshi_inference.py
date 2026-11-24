"""Simulate Moshi inference flow (Mimi -> LMGen -> output tokens) using repo classes.

This script instantiates a small `LMModel` and `LMGen` (with reduced sizes for speed),
then simulates streaming user codes (as Mimi outputs) and shows how the multi-stream
mechanics (cache, delays, init, outputs) operate step-by-step.

It does NOT load checkpoints or real Mimi; instead it demonstrates the exact
inference control flow used in `moshi/moshi/server.py` and `LMGen._step`.

Run with:
  ./.venv/bin/python scripts/simulate_moshi_inference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "moshi"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from moshi.models.lm import LMModel, LMGen
from moshi.modules.transformer import set_attention_context


def main() -> None:
    device = torch.device("cpu")

    # Small model for fast simulation. Keep semantics identical to real code.
    # n_q = number of audio codebooks; dep_q = number of codebooks predicted by Depformer
    n_q = 4
    dep_q = 2
    # num_codebooks = n_q + 1 (text + audio codebooks)
    # Provide 'delays' for each codebook (length = num_codebooks).
    delays = [0, 1, 2, 1, 2]  # example: text delay 0, some audio delays >0

    # Instantiate a lightweight LMModel.
    lm = LMModel(
        delays=delays,
        n_q=n_q,
        dep_q=dep_q,
        card=64,
        text_card=32,
        dim=32,
        num_heads=4,
        num_layers=2,
        hidden_scale=2,
        context=8,
        depformer_dim=32,
        depformer_num_layers=1,
        depformer_num_heads=4,
        device=device,
    )

    lm.to(device)
    lm.eval()
    set_attention_context(lm.depformer, 8)

    # LMGen wraps the LMModel and implements the streaming step logic.
    lm_gen = LMGen(lm_model=lm, use_sampling=False)

    B = 1
    # How many user tokens are expected per step by LMGen._step?
    needed_tokens = lm.num_codebooks - lm.dep_q - 1
    print(f"Model num_codebooks={lm.num_codebooks}, dep_q={lm.dep_q}, needed_tokens(from user)={needed_tokens}")

    # Enter streaming mode (initializes state objects and caches)
    with lm_gen.streaming(B):
        print("Entered streaming mode. Initial streaming state:")
        state = lm_gen._streaming_state
        # show cache shape and initial values
        print(" state.cache.shape =", state.cache.shape)
        print(" state.initial =", state.initial.tolist())
        print(" state.offsets =", state.offsets.tolist())
        print(" delays =", delays)
        print(" CT =", state.cache.shape[2])
        print()

        # Simulate several incoming Mimi frames (each frame -> user provides needed_tokens)
        T = 8
        for step in range(T):
            # Simulate Mimi encoded user tokens for this frame (integers in [0, card))
            user_tokens = torch.randint(0, lm.card, (B, needed_tokens, 1), dtype=torch.long)
            print(f"Step:{step}, user_token = {user_tokens}")
            # Call LMGen.step as server does: one step per time slice
            out = lm_gen.step(user_tokens)

            # Print internal state after write but before next increment (inspect cache & offsets)
            st = lm_gen._streaming_state
            print(f"--- step {step} ---")
            print(" offsets (cpu) =", st.offset_cpu, " offsets(tensor)=", st.offsets.tolist())
            # show a small view of the cache (convert to python lists)
            cache_view = st.cache.cpu().numpy()
            print(" cache (per codebook, indices 0..min(5,CT-1)):")
            for k in range(min(6, cache_view.shape[1])):
                row = cache_view[0, k, :min(6, cache_view.shape[2])].tolist()
                print(f"  k{k}: {row}")

            if out is None:
                print("  out=None (still warming up; waiting for enough frames)")
            else:
                # out shape: [B, dep_q+1, 1] as in server.decode_and_send
                print("  out.shape:", out.shape)
                text_token = out[0, 0, 0].item()
                audio_tokens = out[0, 1:, 0].tolist()
                print(f"  Generated text token (inner monologue) id = {text_token}")
                print(f"  Generated Moshi audio tokens (ids) = {audio_tokens}")
            print()


if __name__ == "__main__":
    main()
