"""End-to-end Moshi inference demo.

This script mirrors the server inference flow:

1. Start from a waveform chunk shaped like server input: [B, 1, frame_size].
2. Run Mimi encode to get discrete codes shaped [B, K, T].
3. Feed each time step into LMGen.step(codes[:, :, t:t+1]).
4. Split the output into text token + audio tokens.
5. Decode audio tokens back with Mimi.decode(tokens[:, 1:]).

The model sizes here are intentionally small for CPU-friendly demonstration,
but the tensor flow matches the real code path in moshi/server.py.

Run:
  ./.venv/bin/python scripts/simulate_moshi_audio_flow.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "moshi"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from moshi.models import loaders
from moshi.models.lm import LMGen, LMModel
from moshi.modules.transformer import set_attention_context


def make_demo_audio(frame_size: int, sample_rate: int, num_frames: int = 4) -> torch.Tensor:
    """Build a synthetic audio chunk shaped [1, 1, frame_size * num_frames]."""
    total_samples = frame_size * num_frames
    t = torch.arange(total_samples, dtype=torch.float32) / sample_rate
    waveform = (
        0.55 * torch.sin(2 * math.pi * 220.0 * t)
        + 0.25 * torch.sin(2 * math.pi * 440.0 * t)
        + 0.10 * torch.sin(2 * math.pi * 660.0 * t)
    )
    waveform = waveform.clamp(-1.0, 1.0)
    return waveform[None, None, :]


def build_demo_lm(device: torch.device) -> LMModel:
    """Small LM that still has the same multi-stream shape logic as Moshi."""
    lm = LMModel(
        delays=[0, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 3, 3, 4],
        n_q=16,
        dep_q=8,
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
    lm.eval()
    set_attention_context(lm.depformer, 8)
    return lm


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")

    print("Loading Mimi architecture (random weights, no checkpoint)...")
    mimi = loaders.get_mimi(None, device=device, num_codebooks=8)
    mimi.eval()

    print("Building small LM for the demo...")
    lm = build_demo_lm(device)
    lm_gen = LMGen(lm_model=lm, use_sampling=False)

    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    audio = make_demo_audio(frame_size=frame_size, sample_rate=mimi.sample_rate, num_frames=6).to(device)

    print()
    print("INPUT")
    print(" waveform shape:", tuple(audio.shape))
    print(" frame_size:", frame_size)
    print(" mimi sample_rate/frame_rate:", mimi.sample_rate, mimi.frame_rate)

    with torch.no_grad():
        codes = mimi.encode(audio)

    print()
    print("MIMI ENCODE")
    print(" codes shape:", tuple(codes.shape))
    print(" active Mimi codebooks:", mimi.num_codebooks)
    print(" first timestep codes:", codes[0, :, 0].tolist())

    needed_tokens = lm.num_codebooks - lm.dep_q - 1
    print()
    print("LM CONFIG")
    print(" num_codebooks:", lm.num_codebooks)
    print(" dep_q:", lm.dep_q)
    print(" needed_tokens from user stream:", needed_tokens)
    print(" LM expects input step shape [B,", needed_tokens, ", 1]")

    print()
    print("STREAMING")
    with lm_gen.streaming(1):
        state = lm_gen._streaming_state
        assert state is not None
        print(" initial cache shape:", tuple(state.cache.shape))
        print(" initial offsets:", state.offsets.tolist())
        print(" initial offsets:", state.offsets.tolist())
        print()

        for t in range(codes.shape[-1]):
            step_codes = codes[:, :, t : t + 1]
            print(f"--- frame {t} ---")
            print(" input to LMGen.step:", tuple(step_codes.shape))
            print(" input codes:", step_codes[0, :, 0].tolist())

            out = lm_gen.step(step_codes)
            st = lm_gen._streaming_state
            assert st is not None
            print(" offsets after step:", st.offsets.tolist())

            if out is None:
                print(" output: None (warming up / delayed)")
            else:
                print(" output shape:", tuple(out.shape))
                text_token = int(out[0, 0, 0].item())
                audio_tokens = out[0, 1:, 0]
                print(" text token (inner monologue):", text_token)
                print(" audio tokens for Mimi.decode:", audio_tokens.tolist())

                with torch.no_grad():
                    reconstructed = mimi.decode(out[:, 1:])
                print(" decoded audio shape:", tuple(reconstructed.shape))

            cache_view = st.cache[0, :, : min(6, st.cache.shape[-1])].tolist()
            print(" cache snapshot (first 6 slots):")
            num_cols = len(cache_view[0]) if cache_view else 0
            col_header = " ".join(f"{col:>4}" for col in range(num_cols))
            print(f"{'stream':>10}  {col_header}")
            for idx, row in enumerate(cache_view):
                formatted_row = " ".join(f"{value:>4}" for value in row)
                print(f"{idx:>10}  {formatted_row}")
            print()


if __name__ == "__main__":
    main()