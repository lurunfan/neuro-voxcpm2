# Low-memory experiment reproduction

The inference adapter uses the standard VoxCPM2 loading interface. Only training with `training/train_lowmem.py` requires the included upstream patch.

## Prepare the upstream source

```powershell
git clone https://github.com/OpenBMB/VoxCPM.git vendor/VoxCPM
git -C vendor/VoxCPM checkout 4b8dd4440d33396b3f429201f8af0529b4d2c601
git -C vendor/VoxCPM apply --check ../../patches/lowmem.patch
git -C vendor/VoxCPM apply ../../patches/lowmem.patch
.\.venv\Scripts\python.exe -m pip install --no-deps -e vendor/VoxCPM
```

Patch only a clean checkout of the pinned source. It constructs frozen weights in BF16, retains FP32 LoRA parameters, disables unused training KV caches, and enables non-reentrant gradient checkpointing. It does not change the inference model architecture.

Download the complete `openbmb/VoxCPM2` model to a local directory such as `models/VoxCPM2` before training. Provide your own audio and two UTF-8 JSONL manifests in a dataset directory:

```json
{"audio":"clips/example.wav","text":"The exact words spoken in this clip.","duration":4.2}
```

Required filenames: `train.candidate.jsonl` and `validation.candidate.jsonl`. Audio paths are relative to the dataset directory. This script expects mono audio and resamples to the AudioVAE input rate. Keep validation recordings separate from the training recordings and listen to both sets before relying on their labels.

```powershell
.\.venv\Scripts\python.exe training/train_lowmem.py --base-model models/VoxCPM2 --dataset data/my_voice --mode probe --output probe_run
.\.venv\Scripts\python.exe training/train_lowmem.py --base-model models/VoxCPM2 --dataset data/my_voice --mode train --steps 48 --accum 8 --lr 0.0001 --output pilot48
```

Training starts from the base model on each run; this is not a resume command. A completed result directory is rejected. A process interrupted before `result.json` exists should be restarted with a new output directory. The probe tests short/medium/long samples, finite nonzero gradients, LoRA updates, a frozen-weight sample, and adapter save/reload equality. GPU training requires CUDA with BF16 support.

Checkpoint folders are saved every 16 updates and at the end. Each contains an optimizer state for inspection; the published inference adapter intentionally contains only the weights, config and checksum, not an optimizer. Dataset caching scales with total audio duration. The reported 4.82 GiB allocation was for this small experiment, not a universal memory bound.

Exact recreation of the published checkpoint requires the original training recordings and labels, which are not included. These steps reproduce the method on user-supplied data; they do not promise bit-identical reconstruction of the published weights.
