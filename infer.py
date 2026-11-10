"""Portable inference for the experimental Neuro-style VoxCPM2 LoRA."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STYLES = {
    'composed': '语气平静自然，发声放松，清晰从容地解释，停顿自然。',
    'natural': '自然聊天，带一点好奇和轻微惊讶，关键词重读，疑问句上扬，语气有起伏。',
    'playful': '轻松而机灵的聊天语气，略带调侃，语速有快有慢，停顿和重音清晰。',
    'prompt': '',
}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def arguments(argv=None):
    p = argparse.ArgumentParser(description='Unofficial Neuro-style VoxCPM2 LoRA / AI 合成')
    p.add_argument('--text')
    p.add_argument('--model', default='openbmb/VoxCPM2', help='HF model ID or local base-model directory')
    p.add_argument('--adapter', type=Path, default=ROOT/'adapters/neuro32')
    p.add_argument('--reference', type=Path, help='User-supplied reference recording')
    p.add_argument('--prompt-text', help='Exact transcript of the reference recording')
    p.add_argument('--style', choices=['auto', *STYLES], default='auto')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path)
    p.add_argument('--local-files-only', action='store_true')
    a = p.parse_args(argv)
    if a.reference and not a.reference.is_file(): p.error('Reference file does not exist')
    if a.prompt_text and not a.reference: p.error('--prompt-text requires --reference')
    if a.style == 'auto': a.style = 'prompt' if a.prompt_text and a.reference else 'composed'
    if a.style == 'prompt' and not (a.reference and a.prompt_text):
        p.error('prompt style requires --reference and --prompt-text')
    a.text = (a.text if a.text is not None else input('请输入要合成的文字：')).strip()
    if not a.text: p.error('Text cannot be empty')
    a.output = (a.output or ROOT/'outputs'/f'{datetime.now():%Y%m%d_%H%M%S_%f}_AI.wav').resolve()
    if a.output.exists() or a.output.with_suffix('.json').exists(): p.error('Output already exists')
    return a

def main():
    a = arguments()
    weights = a.adapter/'lora_weights.safetensors'
    manifest = a.adapter/'manifest.json'
    if manifest.exists():
        expected = json.loads(manifest.read_text(encoding='utf-8'))['sha256']
        if sha256(weights) != expected: raise ValueError('Adapter SHA256 mismatch; download again')
    import numpy as np
    import soundfile as sf
    import torch
    from voxcpm import VoxCPM
    torch.set_num_threads(4)
    model = VoxCPM.from_pretrained(a.model, device=a.device, optimize=False,
        load_denoiser=False, local_files_only=a.local_files_only, lora_weights_path=str(a.adapter))
    loaded, skipped = model.load_lora(str(a.adapter))
    if len(loaded) != 384 or skipped:
        raise RuntimeError(f'Unexpected adapter load: {len(loaded)} loaded, {len(skipped)} skipped')
    control = STYLES[a.style]
    if a.style == 'composed' and not any('\u4e00' <= c <= '\u9fff' for c in a.text): control = ''
    kwargs = dict(text=f'({control}){a.text}' if control else a.text,
        reference_wav_path=str(a.reference) if a.reference else None,
        cfg_value=1.8, inference_timesteps=25 if a.style=='prompt' else 15,
        seed=a.seed, max_len=700, normalize=False, denoise=False, retry_badcase_max_times=1)
    if a.style == 'prompt': kwargs.update(prompt_wav_path=str(a.reference), prompt_text=a.prompt_text)
    audio = np.asarray(model.generate(**kwargs), dtype=np.float32)
    if audio.ndim != 1 or len(audio) < 4800 or not np.isfinite(audio).all() or np.abs(audio).max() < .001:
        raise RuntimeError('Generated audio is empty or invalid')
    gain = min(1., .98 / float(np.abs(audio).max()))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(a.output, audio * gain, 48000, subtype='PCM_16')
    metadata = dict(text=a.text, synthetic=True, model=a.model, adapter=str(a.adapter),
        adapter_sha256=sha256(weights), reference=str(a.reference) if a.reference else None,
        prompt_text=a.prompt_text if a.style=='prompt' else None, style=a.style, control=control,
        cfg=1.8, flow_steps=kwargs['inference_timesteps'], seed=a.seed, lora_strength=1.0,
        loaded_parameters=len(loaded), duration=len(audio)/48000, sample_rate=48000, gain=gain)
    a.output.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'AI audio saved: {a.output}')

if __name__ == '__main__': main()
