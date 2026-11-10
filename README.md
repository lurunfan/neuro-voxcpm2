# Neuro-style VoxCPM2 LoRA

一个非官方的 Neuro 风格语音实验，公开推理代码、低显存训练方法，以及本地试听后选用的 **32 步 LoRA 权重**。基于 [OpenBMB VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)，与 Vedal、Neuro-sama 无隶属或授权背书关系，也不是 Neuro 原项目的语音引擎。

## 仓库包含什么

- `adapters/neuro32/`：36,221,200 字节的真实 safetensors 权重、适配器配置和 SHA256，384 个 LoRA 参数张量。
- `infer.py`：中文/英文等文本推理；默认 CFG 1.8、种子 42、LoRA 强度 1.0，输出 48 kHz WAV 和参数 JSON。
- `training/train_lowmem.py` 与 `patches/lowmem.patch`：BF16 冻结主干、FP32 LoRA、梯度检查点和预缓存音频特征的实验训练实现。
- `docs/`：模型卡、训练结果及复现步骤。

不包含基座权重、原始直播音轨或训练音频。LoRA 不能脱离 VoxCPM2 基座独立运行。

## 安装

本地验证环境：Windows、Python 3.11、RTX 4060 Laptop 8 GB、PyTorch 2.8.0 + CUDA 12.8。以下为 PowerShell 示例，其他系统需使用对应的虚拟环境激活方式和 PyTorch 版本。

```powershell
git clone https://github.com/lurunfan/neuro-voxcpm2.git
cd neuro-voxcpm2
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

音频读取还需要可用的 FFmpeg。安装后确保 `ffmpeg -version` 正常。默认从 Hugging Face 获取 `openbmb/VoxCPM2`，也可通过 `--model` 指定完整本地模型目录。

## 合成语音

不提供参考音频时，可直接试听 LoRA 本身学到的声线：

```powershell
.\.venv\Scripts\python.exe infer.py --text '先别着急，我们可以一步一步来。'
```

也可双击 `Neuro.cmd` 后输入文字。输出在 `outputs/`，文件名带 `AI` 标识。

此前本地选用的效果还使用了**参考录音和准确转写的续接引导**。原始录音未随本仓库分发，因此不带参考的结果不等同于此前选中的演示。提供你有权使用的参考音频及其原文后，可使用同样的参数路径：

```powershell
.\.venv\Scripts\python.exe infer.py --reference references\voice.wav --prompt-text 'Exact words spoken in your reference recording.' --style prompt --text '你也发现了吗？我刚才还以为，是我看错了。'
```

`prompt` 模式采用 25 个推理步数；`composed`、`natural`、`playful` 模式采用 15 步。32 是适配器的训练更新步数，与推理步数不是同一个概念。`--style auto` 在同时给出参考和转写时选择 prompt，否则选择 composed。

支持 `--model`、`--adapter`、`--reference`、`--prompt-text`、`--style`、`--device`、`--seed`、`--output` 和 `--local-files-only`。现有输出文件不会被覆盖，载入发布权重前会检查 SHA256。

## 训练结果与限制

实际实验使用 170 条训练候选、16 条独立时段验证候选，全部候选约 15 分 29 秒。原始候选为自动筛选，未完成人工逐句听审。48 次更新后验证损失从 1.004559 降至 0.920838；32 步约为 0.926000。用户试听偏好 32 步，因此发布 32 步，不能以较低损失断言 48 步更自然。

训练循环的 PyTorch 分配显存峰值约 4.82 GiB，不代表整张显卡总占用，也不是所有训练配置的最低显存承诺。说话人相似度、中文自然度、长句稳定性没有正式基准评估。可能出现朗读感、杂音、重复或发音差异。随机种子也不保证跨硬件逐字节一致。

来源与详细限制见 [模型卡](docs/MODEL_CARD.md)，训练方式见 [训练说明](docs/TRAINING.md)。

## 检查与许可

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

代码和本仓库发布的适配器采用 Apache-2.0，见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。基座模型按其上游许可提供。该许可不授予第三方原始录音、角色、名称或商标的权利。请明确标注 AI 合成，不要将声音伪称为官方或真人原始录音。
