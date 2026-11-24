# Moshi, one audio frame at a time

Open the [Moshi Audio Explorer](explorer.html) in your browser to play the audio, scrub through 80 ms frames, and inspect each stage of the recorded run. The HTML includes the audio and trace data for offline use.

Each stage has a **Component flow** diagram. The nodes show tensor shapes and IDs for the selected frame, including the semantic and acoustic branches of Mimi. In **Depformer**, click a codebook to see what conditions that prediction. In **Align streams**, inspect the two calls on frame 0 to see why the first call fills the ring buffer before the second emits the aligned output.

In **Temporal LM**, the **Inner Monologue** shows the selected text ID and its SentencePiece vocabulary piece alongside the top candidates. **Align streams** shows the piece for the emitted text ID, which can come from an earlier LM call. For example, `14166` maps to `▁cascade`, where `▁` represents a leading space. The explorer embeds the pieces it needs from [Kyutai’s text tokenizer](https://huggingface.co/kyutai/moshiko-pytorch-bf16/blob/2bfc9ae/tokenizer_spm_32k_3.model). Looking up existing IDs requires only the tokenizer. Generating a meaningful reply requires trained language-model weights.

Listen to [the input](sample.wav), an eSpeak NG recording of “Hello Moshi. What is two plus two?” Then open the [architecture diagram](architecture.html) and follow the [terminal trace](run/terminal.txt). Each trace line shows a real operation from the PyTorch implementation. The [JSON trace](run/trace.json) contains the same kinds of details for every frame.

The run uses Kyutai’s **pretrained Mimi codec** and a **small randomly initialized Moshi language model**. The trained codec reconstructs the input. The toy model demonstrates token prediction and streaming alignment, and its output is noise or speech-like sounds without a meaningful answer. This machine has no CUDA device or enough available RAM for the [15.4 GB Moshiko BF16 checkpoint](https://huggingface.co/kyutai/moshiko-pytorch-bf16/tree/main). The same tracing script accepts the original BF16 checkpoint and tokenizer.

## What travels through the model

| Stage | Per-frame result | What changes |
| --- | --- | --- |
| Input | `[1, 1, 1920]` float PCM | One batch, one channel, 80 ms at 24 kHz |
| Mimi SEANet encoder | `[1, 512, 2]` | Causal convolutions turn 1,920 samples into two 512-channel feature vectors |
| Mimi encoder Transformer | `[1, 512, 2]` | Streaming attention updates those two vectors using recent encoder history |
| Mimi downsample | `[1, 512, 1]` | A learned strided convolution reduces 25 positions/s to 12.5 positions/s |
| Mimi split RVQ | `[1, 8, 1]` integer IDs | The semantic and acoustic branches project into 256 channels. One semantic codebook and seven acoustic codebooks produce eight IDs from separate tables of 2,048 entries each |
| LMGen and Temporal Transformer | `[1, 17, 1]` IDs → `[1, 1, 64]` hidden state | LMGen builds the delayed input. The model sums embeddings from all streams and uses causal attention over prior steps. The trained 7B model’s hidden width is 4,096 |
| Text head | `[1, 1, 1, 32000]` logits → one ID | The model predicts its own text stream, including padding steps. The trained checkpoint’s SentencePiece tokenizer renders its text pieces |
| Depformer | Eight calls, each with `[1, 1, 1, 2048]` logits | It predicts semantic audio ID 0 from the temporal state and text ID. Each next audio ID uses that state and the preceding audio ID |
| LMGen output | `[1, 9, 1]` aligned IDs | The ring buffer releases one text ID and eight generated audio IDs. The trace records predicted and emitted IDs separately |
| Mimi decoder | `[1, 8, 1]` IDs → `[1, 1, 1920]` PCM | Codebook lookup and projection rebuild a feature vector. Upsampling, decoder attention, and SEANet synthesize the next waveform chunk |

Read the implementation in [Mimi encode/decode](../../moshi/moshi/models/compression.py#L338), [Mimi split RVQ](../../moshi/moshi/quantization/vq.py#L269), [Temporal Transformer inputs and text head](../../moshi/moshi/models/lm.py#L379), [LMGen alignment](../../moshi/moshi/models/lm.py#L669), and [Depformer generation](../../moshi/moshi/models/lm.py#L809). The runnable reference loop is [offline inference](../../moshi/moshi/run_inference.py#L140). The [server](../../moshi/moshi/server.py#L130) adds Opus I/O and its own session startup.

### How to read the first two LM calls

- There are 17 stream rows in the token ring buffer. Row 0 is Moshi text, rows 1–8 are Moshi audio and rows 9–16 are user audio. The three columns are reused cyclically. LMGen writes its predictions back into this buffer so later temporal steps can consume them alongside incoming user IDs. Attention stores its own key/value tensors in a separate cache.
- The first LM call sees begin-of-stream tokens. The audio start ID is `2048`, and the text start ID is `32000`. Unwritten ring-buffer cells are marked `-2`. The model predicts tokens on this call and delays emission while the streams warm up.
- The offline runner feeds the first input codes again on the next LM call. With the default delay pattern, text and each semantic audio stream have delay 0. Each acoustic stream has delay 1. When constructing the emitted frame, LMGen selects the aligned streams from their respective ring slots. Compare the first aligned text ID with the two raw text predictions, then do the same for audio codebooks 0 and 1.
- The temporal attention state persists across frames. The Depformer starts a fresh streaming context each LM step and keeps that context through the eight codebook predictions. In the trace, `first_layer_kv.shape` gives `[key/value, batch, heads, capacity, head dimension]`, and `end_offset` counts how many positions that cache has processed since it was created.
- The trace prints the five highest scores after softmax for each prediction. These are computed from the unmodified logits. With `--sample`, the repository sampler applies its temperature and top-k settings separately.

Frame 0 encoded to the user IDs `[1049, 1638, 920, 328, 2025, 2047, 1272, 1720]`. The initial LM call warmed up the stream. The next call caused LMGen to emit a frame using the first call’s text and semantic prediction and the next call’s acoustic predictions.

| Frame 0 output | Text ID | Semantic ID | Seven acoustic IDs |
| --- | ---: | ---: | --- |
| First LM call, raw | 14166 | 698 | 967, 1397, 402, 1580, 897, 909, 253 |
| Second LM call, raw | 12263 | 116 | 1040, 1397, 402, 1580, 462, 1017, 162 |
| Emitted frame | **14166** | **698** | **1040, 1397, 402, 1580, 462, 1017, 162** |

The run processed 28 input frames and 25 silence frames. It made 54 LM calls, including warm-up, and emitted 53 audio frames. The Mimi file matched the published checkpoint SHA-256 `09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50`.

### Listen to the codec and the generation path

1. [Input sent to Mimi](run/input.wav)
2. [Mimi reconstruction, one semantic codebook](run/mimi-reconstruction-1cb.wav)
3. [Mimi reconstruction, all eight codebooks](run/mimi-reconstruction-8cb.wav)
4. [Toy model output decoded by Mimi](run/toy-lm-output.wav)

The two reconstructions decode **the input IDs** in fresh streaming states. Listen for what adding seven acoustic codebooks changes. The fourth file decodes **new IDs generated by the toy LM**. In a trained conversation, the user’s eight audio streams keep coming in as the model generates its own text and audio streams. The generated text conditions audio generation inside each step. Incoming speech enters as Mimi IDs. At 12.5 Hz, eight 11-bit codebook IDs represent 1,100 bits/s of raw codec IDs before packing overhead.

An ID is an index into one codebook. The numeric difference between two IDs has no acoustic interpretation by itself. During decoding, Mimi looks up the vectors for those IDs and adds the projected semantic and acoustic contributions.

The eight-codebook reconstruction follows the original speech bursts and pauses. In this sample the one-codebook waveform has a much flatter amplitude envelope. [Compare the four waveforms on an 80 ms grid](waveforms.png).

## Reproduce the trace

From the repository root. The dependencies installed for the recorded run live in `/tmp/moshi-trace-deps` and leave the Conda environment untouched. If the temporary directory is gone, recreate it with the first command.

```bash
conda activate data_viz
python -m pip install --target /tmp/moshi-trace-deps 'einops>=0.7,<0.9' 'sentencepiece>=0.2,<0.3' 'sphn>=0.2,<0.3'

curl -fsSL --retry 2 \
  https://huggingface.co/kyutai/moshiko-pytorch-bf16/resolve/main/tokenizer-e351c8d8-checkpoint125.safetensors \
  -o /tmp/tokenizer-e351c8d8-checkpoint125.safetensors

PYTHONPATH="/tmp/moshi-trace-deps${PYTHONPATH:+:$PYTHONPATH}" \
python scripts/trace_moshi_audio.py \
  --mimi /tmp/tokenizer-e351c8d8-checkpoint125.safetensors \
  --output /tmp/moshi-trace-run \
  --print-frames 3
```

Pass `--input path/to/audio.wav` to trace your own audio. Input is converted to 24 kHz mono, zero-padded to a whole frame when necessary, and followed by 25 silence frames (2 seconds) by default. Use `--tail-frames` to change how long the stream continues. Use `--print-frames 53` to print every stage of every frame in the bundled sample. IDs are logged in full. Floating-point activations include their shape, first six values, RMS, and range.

For a pretrained spoken reply and the corresponding offline HTML, follow the [A100 tracing guide](A100.md).

## Diagram verification

The data-flow diagram passed 9 of 9 Archify showcase checks with zero errors and warnings. Automated browser checks passed at four desktop sizes from 1440 × 900 to 2048 × 1320. I inspected the rendered light and dark layouts.

| File | SHA-256 |
| --- | --- |
| [Diagram specification](architecture.json) | `9fc2b144eb402bfad6ee62f186e3cda754ee6b053ec1b66becd0a793ea7606eb` |
| [Standalone diagram](architecture.html) | `1423496b6a9808b6ab51dba60bfe5470d1b986049e08fdeebeeb570f313d33da` |
