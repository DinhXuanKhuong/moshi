"""Trace the local PyTorch Mimi → LMGen → Mimi streaming path."""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from functools import wraps
from itertools import chain, repeat
from pathlib import Path
from unittest.mock import patch

import sentencepiece
import sphn
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "moshi"))

from moshi.models import loaders
from moshi.models.lm import LMGen
from moshi.utils.compile import no_compile, no_cuda_graph


def summarize(value: torch.Tensor) -> dict:
    flat = value.detach().flatten()
    summary = {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch.")}
    if value.is_floating_point():
        samples = flat.float()
        summary.update(
            first=[round(number, 6) for number in samples[:6].tolist()],
            rms=round(samples.square().mean().sqrt().item(), 6),
            min=round(samples.min().item(), 6),
            max=round(samples.max().item(), 6),
        )
    else:
        summary["values"] = flat.tolist()
    return summary


def top_scores(logits: torch.Tensor) -> list[dict]:
    probabilities = logits[0, 0, 0].float().softmax(dim=-1)
    values, indices = probabilities.topk(5)
    return [{"id": index, "p": round(value, 7)} for index, value in zip(indices.tolist(), values.tolist())]


def first_layer_kv(transformer) -> dict:
    state = transformer.layers[0].self_attn._streaming_state
    assert state is not None and state.kv_cache is not None
    cache = state.kv_cache
    return {"shape": list(cache.cache.shape), "end_offset": cache.end_offset.item()}


def trace_module(stack: ExitStack, module, name: str, record) -> None:
    def hook(_, inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        record(name, input=summarize(inputs[0]), output=summarize(output))

    handle = module.register_forward_hook(hook)
    stack.callback(handle.remove)


def trace_method(stack: ExitStack, owner, name: str, callback) -> None:
    original = getattr(owner, name)

    @wraps(original)
    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        callback(args, result)
        return result

    stack.enter_context(patch.object(owner, name, wrapped))


def write_audio(path: Path, chunks: list[torch.Tensor], sample_rate: int) -> None:
    pcm = torch.cat(chunks, dim=-1)[0, 0].cpu().float().clamp(-1, 1)
    sphn.write_wav(str(path), pcm.numpy(), sample_rate)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "demos/moshi-audio-trace/sample.wav")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mimi", type=Path, required=True, help="Moshiko Mimi .safetensors")
    parser.add_argument("--moshi", type=Path, help="Optional original Moshiko BF16 model.safetensors")
    parser.add_argument("--tokenizer", type=Path, help="Moshiko SentencePiece .model, required with --moshi")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--tail-frames", type=int, default=25, help="Silence steps after the input")
    parser.add_argument("--print-frames", type=int, default=3, help="Print detailed stages for these first frames")
    parser.add_argument("--sample", action="store_true", help="Use LMGen top-k sampling instead of argmax")
    args = parser.parse_args()
    if bool(args.moshi) != bool(args.tokenizer):
        parser.error("--moshi and --tokenizer must be supplied together")
    if args.tail_frames < 0 or args.print_frames < 0 or args.threads < 1:
        parser.error("frame counts must be nonnegative and threads must be positive")

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable. Check the NVIDIA driver and PyTorch installation.")
        with torch.cuda.device(device):
            if args.moshi and not torch.cuda.is_bf16_supported(including_emulation=False):
                parser.error("Pretrained CUDA tracing requires a GPU with native BF16 support.")
            print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    print(f"Loading pretrained Mimi on {device}...", flush=True)
    mimi = loaders.get_mimi(args.mimi, device=device)
    if args.moshi:
        lm = loaders.get_moshi_lm(args.moshi, device=device)
        tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer))
        mode = "pretrained Moshiko"
    else:
        lm = loaders.get_moshi_lm(
            None, device=device, dtype=torch.float32,
            lm_kwargs_overrides={
                "dim": 64, "num_layers": 2, "num_heads": 4, "context": 32,
                "depformer_dim": 32, "depformer_num_heads": 4, "depformer_num_layers": 1,
                "depformer_dim_feedforward": 64,
            },
        )
        tokenizer = None
        mode = "random-weight miniature Moshi"
    print(f"Language model: {mode}", flush=True)
    sample_rate, frame_size = mimi.sample_rate, mimi.frame_size
    pcm, _ = sphn.read(str(args.input), sample_rate=sample_rate)
    audio = torch.from_numpy(pcm).float().mean(dim=0, keepdim=True).unsqueeze(0).to(device)
    if audio.shape[-1] == 0:
        parser.error("input must contain audio")
    source_samples = audio.shape[-1]
    audio = torch.nn.functional.pad(audio, (0, (-source_samples) % frame_size))
    frames = audio.split(frame_size, dim=-1)
    silence = torch.zeros_like(frames[0])
    input_codes = []
    emitted_pcm = []
    results = []

    def record(stage: str, **details) -> None:
        events.append({"stage": stage, **details})
        if frame_index < args.print_frames:
            print(f"  {stage} {json.dumps(details)}")

    lm_gen = LMGen(
        lm, use_sampling=args.sample, check=True,
        on_text_hook=lambda token: record("Text ID before alignment", id=token.item()),
        on_audio_hook=lambda tokens: record("Audio IDs before alignment", ids=tokens[0].tolist()),
    )
    trace = {
        "model": mode, "sampling": "top-k" if args.sample else "argmax", "seed": 0,
        "device": str(device), "torch_version": torch.__version__,
        "mimi_checkpoint": args.mimi.name, "moshi_checkpoint": args.moshi.name if args.moshi else None,
        "input_name": args.input.name,
        "sample_rate": sample_rate, "frame_size": frame_size, "source_samples": source_samples,
        "audio_codebooks": mimi.num_codebooks, "audio_vocabulary": mimi.cardinality,
        "delays": lm.delays, "audio_frames": len(frames), "tail_frames": args.tail_frames,
        "frames": results,
    }

    with no_compile(), no_cuda_graph():
        with ExitStack() as stack:
            for name, module in [
                ("Mimi SEANet encoder", mimi.encoder),
                ("Mimi encoder Transformer", mimi.encoder_transformer),
                ("Mimi downsample", mimi.downsample),
                ("Semantic RVQ input projection", mimi.quantizer.rvq_first.input_proj),
                ("Acoustic RVQ input projection", mimi.quantizer.rvq_rest.input_proj),
                ("Summed embeddings → temporal Transformer", lm.transformer),
                ("Temporal state → text head", lm.text_linear),
                ("Depformer Transformer", lm.depformer),
                ("Mimi upsample", mimi.upsample),
                ("Mimi decoder Transformer", mimi.decoder_transformer),
                ("Mimi SEANet decoder", mimi.decoder),
            ]:
                trace_module(stack, module, name, record)
            for index, layer in enumerate(lm.transformer.layers):
                trace_module(stack, layer, f"Temporal layer {index}", record)
            trace_method(stack, mimi.quantizer, "encode", lambda inputs, codes: record(
                "Split RVQ encode", input=summarize(inputs[0]), output=summarize(codes)))
            trace_method(stack, mimi.quantizer, "decode", lambda inputs, latent: record(
                "Split RVQ decode", input=summarize(inputs[0]), output=summarize(latent)))
            trace_method(stack, lm, "forward_text", lambda inputs, result: record(
                "Temporal prediction", delayed_inputs=summarize(inputs[0]),
                hidden=summarize(result[0]), top_text=top_scores(result[1]),
                first_layer_kv=first_layer_kv(lm.transformer)))
            trace_method(stack, lm, "forward_depformer", lambda inputs, logits: record(
                "Depformer prediction", codebook=inputs[0], previous_id=inputs[1].item(),
                logits_shape=list(logits.shape), top_audio=top_scores(logits),
                first_layer_kv=first_layer_kv(lm.depformer)))
            stack.enter_context(mimi.streaming(1))
            stack.enter_context(lm_gen.streaming(1))
            print(f"{len(frames)} input frames + {args.tail_frames} silence frames, delays={lm.delays}")
            for frame_index, frame in enumerate(chain(frames, repeat(silence, args.tail_frames))):
                events = []
                print(f"Frame {frame_index:02d} | t={frame_index * frame_size / sample_rate:.2f}s", flush=True)
                codes = mimi.encode(frame)
                if frame_index < len(frames):
                    input_codes.append(codes)
                state = lm_gen._streaming_state
                assert state is not None
                # Like run_inference.py, reuse the first input frame after the BOS step.
                offsets = []
                for _ in range(2 if frame_index == 0 else 1):
                    offset = state.offset_cpu
                    emitted = lm_gen.step(codes)
                    offsets.append(offset)
                    record(
                        "LMGen ring buffer", before=offset, after=state.offset_cpu,
                        cache=state.cache[0].tolist(), emitted=emitted is not None,
                    )
                text_id = None
                audio_ids = None
                piece = None
                if emitted is not None:
                    text_id = emitted[0, 0, 0].item()
                    audio_ids = emitted[0, 1:, 0].tolist()
                    if tokenizer and text_id not in (0, lm.text_padding_token_id):
                        piece = tokenizer.id_to_piece(text_id)
                    emitted_pcm.append(mimi.decode(emitted[:, 1:]))
                incoming = codes[0, :, 0].tolist()
                results.append({
                    "index": frame_index, "time_ms": 1000 * frame_index * frame_size / sample_rate,
                    "silence_tail": frame_index >= len(frames), "lm_offsets": offsets,
                    "user_ids": incoming, "text_id": text_id, "piece": piece, "moshi_ids": audio_ids,
                    "events": events,
                })
                print(f"  user={incoming} | aligned text={text_id} {piece or ''} | audio={audio_ids}")

        # Independent decoder states keep the comparison streams from contaminating one another.
        for codebooks in (1, mimi.num_codebooks):
            with mimi.streaming(1):
                reconstruction = [mimi.decode(codes[:, :codebooks]) for codes in input_codes]
            write_audio(output / f"mimi-reconstruction-{codebooks}cb.wav", reconstruction, sample_rate)

    write_audio(output / "input.wav", [audio], sample_rate)
    if emitted_pcm:
        write_audio(output / ("moshiko-reply.wav" if tokenizer else "toy-lm-output.wav"), emitted_pcm, sample_rate)
    trace["text"] = "".join(frame["piece"] or "" for frame in results).replace("▁", " ").strip() if tokenizer else None
    if tokenizer:
        text_ids = {frame["text_id"] for frame in results if frame["text_id"] is not None}
        for frame in results:
            for event in frame["events"]:
                if event["stage"] == "Text ID before alignment":
                    text_ids.add(event["id"])
                elif event["stage"] == "Temporal prediction":
                    text_ids.update(candidate["id"] for candidate in event["top_text"])
        trace["text_pieces"] = {token: tokenizer.id_to_piece(token) for token in sorted(text_ids)}
    trace["emitted_audio_frames"] = len(emitted_pcm)
    (output / "trace.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n")
    if tokenizer:
        print(f"Reply text: {trace['text']}")
    print(f"Saved {len(results)} traced input steps to {output}")


if __name__ == "__main__":
    main()
