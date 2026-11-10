# Model card: Neuro-style VoxCPM2 LoRA (step 32)

- Publisher: lurunfan; unofficial community experiment.
- Base: OpenBMB VoxCPM2, Apache-2.0. Upstream code pinned at `4b8dd4440d33396b3f429201f8af0529b4d2c601`.
- Artifact: `adapters/neuro32/lora_weights.safetensors`; checksum and size in its adjacent `manifest.json`.
- Adapter: rank 16, alpha 16, attention q/k/v/o projections on LM and DiT; 384 tensors.
- Training: 48 optimizer updates, accumulation 8, batch 1, initial learning rate 1e-4 with warm-up/cosine scheduling. This artifact is the checkpoint saved at update 32, not the final 48-update checkpoint.
- Base parameters frozen in BF16, adapters FP32, AudioVAE features cached. Gradient checkpointing and standard save/reload were checked locally.

## Data provenance

The experiment used automatically screened excerpts of an official-channel public replay: [2026-10-07 session](https://www.bilibili.com/video/BV1x3pN6SE1p?p=1), uploaded by Vedal和Neuro-sama. Candidate time windows: 09:00–17:00, 40:00–62:00, 70:00–76:00, 94:00–97:00. The last window supplies held-out validation rather than a random split of adjacent utterances.

283 input segments → 186 candidate clips (170 train / 16 validation), approximately 15.48 minutes total. Screening used VAD, independent SenseVoice/Whisper transcripts, waveform quality checks and deduplication. The data is **machine-screened and not manually auditioned throughout**; this is not proof of single-speaker purity, no background music, or perfect transcription. Raw recordings and full transcripts are not distributed here.

The locally preferred prompt reference was `neuro_20261007_0558960` from that source. Its ID alone is not sufficient to reproduce the audio conditioning: supply a suitable recording and exact transcript yourself. No generated speech from another TTS system was used for this Neuro training run.

## Evaluation

Fixed-seed validation loss: base 1.004559; step 16 0.961341; step 32 about 0.926000; step 48 0.920838. Full numerical record: [training_result.json](training_result.json). User listening preferred step 32 despite the lower numerical loss at 48. No controlled human study, speaker-verification benchmark or broad multilingual evaluation was performed.

The bundled CLI has a no-reference fallback for usability. Previously preferred local samples additionally used original-reference continuation, CFG 1.8, 25 inference steps, seed 42 and adapter strength 1.0. The no-reference fallback is not claimed to reproduce that sound.

## Scope and limitations

Intended for local synthetic-voice experimentation and research into low-memory LoRA training. Audio should be identified as AI-generated. This repository does not contain Neuro's original voice engine and has no official affiliation or endorsement. Similarity and naturalness remain subjective; audio artifacts, text errors and expressiveness problems are possible. Data provenance is documented, but the repository license does not grant rights in the source broadcasts or character identity.
