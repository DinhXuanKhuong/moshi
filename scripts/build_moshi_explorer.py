"""Package a Moshi trace and its four WAVs into the offline HTML explorer."""
from __future__ import annotations

import argparse
import base64
import json
import sys
import wave
from array import array
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "demos/moshi-audio-trace/explorer.html"
DATASET_OPEN = '<script type="application/json" id="dataset">'


def audio_track(path: Path, name: str, color: str) -> dict:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getnframes() == 0:
            raise ValueError(f"{path} must contain mono 16-bit PCM")
        samples = array("h", wav.readframes(wav.getnframes()))
        duration = len(samples) / wav.getframerate()
    if sys.byteorder != "little":
        samples.byteswap()
    block = max(1, (len(samples) + 1199) // 1200)
    peaks = []
    for offset in range(0, len(samples), block):
        chunk = samples[offset:offset + block]
        peaks.append([round(min(chunk) / 32768, 5), round(max(chunk) / 32768, 5)])
    return {
        "name": name, "short": name, "file": path.name, "color": color,
        "duration": duration, "peaks": peaks,
        "audio": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Trace output directory")
    parser.add_argument("--output", type=Path, help="Defaults to RUN/explorer.html")
    args = parser.parse_args()
    output = args.output or args.run / "explorer.html"
    if output.resolve() == TEMPLATE.resolve():
        parser.error("Choose an output path outside the bundled explorer template.")

    template = TEMPLATE.read_text(encoding="utf-8")
    before, marker, rest = template.partition(DATASET_OPEN)
    bundled, end, after = rest.partition("</script>")
    if not marker or not end:
        parser.error("The explorer template is missing its embedded dataset.")
    trace = json.loads((args.run / "trace.json").read_text(encoding="utf-8"))
    text_pieces = trace.get("text_pieces", {})
    pretrained = bool(trace["moshi_checkpoint"])
    if pretrained and not text_pieces:
        parser.error("This checkpoint trace lacks text_pieces. Run it with the updated tracer.")
    if not pretrained and not text_pieces:
        text_pieces = json.loads(bundled).get("textPieces", {})
    tracks = [
        audio_track(args.run / "input.wav", "Source speech", "#ffba78"),
        audio_track(args.run / "mimi-reconstruction-1cb.wav", "Mimi · 1 codebook", "#cbb9ff"),
        audio_track(args.run / "mimi-reconstruction-8cb.wav", "Mimi · 8 codebooks", "#8cdec1"),
        audio_track(
            args.run / ("moshiko-reply.wav" if pretrained else "toy-lm-output.wav"),
            "Moshiko reply" if pretrained else "Toy LM output", "#f5a2bd",
        ),
    ]
    dataset = json.dumps(
        {"trace": trace, "tracks": tracks, "textPieces": text_pieces},
        ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).replace("<", r"\u003c")
    output.write_text(before + marker + dataset + end + after, encoding="utf-8")
    print(f"Offline explorer: {output}")


if __name__ == "__main__":
    main()
