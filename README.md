# Breeze TTS 2 — NF4 deployment

Fork of [`breezeblue-ai/breeze-tts`](https://github.com/breezeblue-ai/breeze-tts) adapted for **bitsandbytes NF4** inference against the official weights at [`BreezeBlue/Breeze-TTS-2`](https://huggingface.co/BreezeBlue/Breeze-TTS-2).

> Source code is Apache 2.0. Breeze TTS 2 weights, derivative models, and self-hosted outputs are research / non-commercial only. See [License](#license-and-responsible-use).

## What changed in this fork

Hybrid 4-bit load path, not a full-model dump:

| Module | Precision | Why |
| --- | --- | --- |
| Text encoder linears | **NF4** (double-quant) | Biggest parameter block, instruction/text path |
| Backbone (`Qwen3`) linears | **NF4** | Biggest VRAM cut |
| Depth decoder | **bf16** (optional NF4) | 15-step codebook decode; keep quality by default |
| Codec / vocoder | **bf16** | Quantizing the waveform net makes speech gritty |
| Embeddings, RMSNorm, `lm_head` | **bf16** | Tiny, quality-sensitive |

CUDA-graph `--fast-*` flags are forced off under `--nf4`. bitsandbytes `Linear4bit` is not safe inside captured graphs.

Expected VRAM (eager, batch 1, unofficial estimate):

| Mode | Weights + activations |
| --- | --- |
| Official eager bf16 | ~7.7 GiB |
| This fork `--nf4` | ~3.5–5 GiB |
| `--nf4 --nf4-include-depth-decoder` | ~3–4 GiB |

A 12 GB card (RTX 3060) is the target. A 24 GB 3090 still benefits if you want headroom for other models.

## Setup

```bash
git clone https://github.com/groxaxo/breeze-tts.git
cd breeze-tts
python -m pip install -r requirements.txt
bash scripts/download_breeze_tts2.sh ./checkpoints/Breeze-TTS-2
```

The snapshot **must** include `audio_tokenizer/`.

## NF4 inference

```bash
python infer.py ./checkpoints/Breeze-TTS-2 \
  --nf4 \
  --text "(sigh) Welcome aboard. Your journey begins now." \
  --instruction "A warm, thoughtful young woman with a clear voice and a calm, reflective delivery." \
  --cfg-scale 4 \
  --output outputs/voice_design_nf4.wav
```

Voice clone:

```bash
python infer.py ./checkpoints/Breeze-TTS-2 \
  --nf4 \
  --ref-audio reference_en.wav \
  --ref-text "This is the exact transcript of the English reference audio." \
  --text "(sigh) It is good to hear your voice again after all this time." \
  --output outputs/voice_clone_nf4.wav
```

Streaming API:

```bash
python -m breeze_infer.api ./checkpoints/Breeze-TTS-2 --nf4 --host 0.0.0.0 --port 7860
```

`GET /health` reports `"nf4": true` when the 4-bit path is loaded.

### Flags

| Flag | Default | Effect |
| --- | --- | --- |
| `--nf4` / `--no-nf4` | off | Quantize text encoder + backbone linears |
| `--nf4-include-depth-decoder` | off | Also quantize the depth decoder |

## Official eager / fast path

Same CLI as upstream. Omit `--nf4` and use `--fast-all` on a 24 GB GPU if you want the CUDA-graph path.

```bash
python infer.py ./checkpoints/Breeze-TTS-2 \
  --text "Hello from Breeze." \
  --output outputs/eager.wav
```

## Files added or modified

- `breeze_infer/nf4.py` — Linear → `bitsandbytes.nn.Linear4bit` walker
- `breeze_infer/runtime.py` — `nf4=` load hook, CPU convert then `.to(device)`
- `infer.py`, `breeze_infer/api.py` — CLI / API flags
- `scripts/download_breeze_tts2.sh` — HF snapshot helper
- `tests/test_nf4.py` — skip-list unit test
- `requirements.txt` — `bitsandbytes`, `accelerate`, `huggingface_hub`

## License and Responsible Use

The source code is licensed under the [Apache License, Version 2.0](LICENSE). The audio tokenizer is based on [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache 2.0). Model weights, checkpoints, adapters, derivative models, and self-hosted outputs are governed by the [BreezeBlue Research and Non-Commercial License](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE). The Apache License does not grant rights to use the model commercially.

Unauthorized voice cloning, impersonation, fraud, and other unlawful uses are prohibited.

Upstream: [breezeblue-ai/breeze-tts](https://github.com/breezeblue-ai/breeze-tts) · Weights: [BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2)
