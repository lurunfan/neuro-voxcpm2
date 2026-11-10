# Release verification (2026-11-10)

- Five CLI/checksum tests passed: required reference handling, blank input rejection, no-reference fallback, refusing an existing output, and detecting changed file bytes.
- All Python entry points compiled, and the training patch passed `git apply --check` against clean copies of the two upstream files at the pinned commit.
- The portable `infer.py` actually loaded all 384 adapter parameters with zero skipped parameters, using the published adapter folder and the local VoxCPM2 base model.
- A no-reference Chinese smoke test produced a finite 48 kHz mono waveform, 2.24 seconds, peak 0.841156. Local SenseVoice transcription matched the target text ignoring punctuation.
- Adapter SHA256: `522a4a78c7c4002fc2326ef3cc7edb6982b216eee2418de009fc3951da5c390c` (36,221,200 bytes), equal to the selected original local step-32 artifact.
- This verification used an existing compatible local environment. A completely fresh installation of the documented requirements was not performed.
- Original broadcasts, local reference recordings, training audio, optimizer states, credentials, private chat material and unrelated voice projects are outside the release file list.

Code review covered path portability, output overwrite protection, reference/transcript validation, adapter integrity, subprocess argument handling in the Windows launcher and consistency between documented and implemented training settings. Machine transcription does not establish voice similarity or naturalness. No new human listening evaluation is claimed.
