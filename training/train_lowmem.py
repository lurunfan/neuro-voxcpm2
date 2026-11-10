"""BF16 frozen weights + FP32 LoRA, cached VAE features, batch size one.

Probe mode verifies real gradients, parameter updates and adapter round-trip.
Candidate data is explicitly experimental; no human audition is implied.
"""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BASE = Path(__file__).resolve().parent
os.environ['HF_HOME'] = str(ROOT/'cache/huggingface')
os.environ['TORCH_HOME'] = str(ROOT/'cache/torch')
os.environ['PATH'] = str(ROOT/'tools/ffmpeg/local/bin')+os.pathsep+os.environ['PATH']
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys
# Install the patched upstream source in editable mode; see docs/TRAINING.md.
import argparse
import gc
import json
import random
import time
import torch
import soundfile as sf
import torchaudio
from safetensors.torch import save_file, load_file
from voxcpm.model.voxcpm2 import VoxCPM2Model, LoRAConfig
from voxcpm.modules.minicpm4.model import MiniCPMModel
from voxcpm.training.data import BatchProcessor, HFVoxCPMDataset

def emit(event, **values):
    print(json.dumps(dict(event=event,**values),ensure_ascii=False),flush=True)

def forward(model,b):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        return model(**{k:b[k].cuda() for k in ['text_tokens','text_mask','audio_feats','audio_mask','loss_mask','position_ids','labels']})

def evaluate(model,batches):
    was_training=model.training
    model.eval()
    try:
        with torch.random.fork_rng(devices=[0]),torch.no_grad():
            torch.manual_seed(123)
            values=[]
            for b in batches:
                r=forward(model,b);values.append(float(r['loss/diff']+r['loss/stop']))
        assert values and all(torch.isfinite(torch.tensor(v)) for v in values)
        return sum(values)/len(values)
    finally:
        model.train(was_training)

def save_adapter(model,folder,step,optimizer):
    folder.mkdir(parents=True,exist_ok=True)
    state={k:p.detach().cpu().contiguous() for k,p in model.named_parameters() if p.requires_grad}
    save_file(state,str(folder/'lora_weights.safetensors'))
    config={'lora_config':model.lora_config.model_dump(),'base_model':model._publication_base_path}
    (folder/'lora_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    torch.save(optimizer.state_dict(),folder/'optimizer.pth')
    (folder/'training_state.json').write_text(json.dumps({'step':step,'experimental_candidate_data':True}),encoding='utf-8')
    loaded=load_file(str(folder/'lora_weights.safetensors'))
    assert state.keys()==loaded.keys() and all(torch.equal(v,loaded[k]) for k,v in state.items())
    return state

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['probe','train'],default='probe')
    parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--accum',type=int,default=8)
    parser.add_argument('--output',default='lowmem_probe')
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--base-model',type=Path,required=True)
    parser.add_argument('--lr',type=float,default=1e-4)
    args=parser.parse_args()
    assert args.steps>0 and args.accum>0
    assert 0<args.lr<=1e-3
    dataset=args.dataset.resolve()
    out=dataset/args.output
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        raise ValueError('Use a new output directory; this run already has a result')
    torch.set_num_threads(4)
    torch.manual_seed(42);random.seed(42)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    entries=[json.loads(x) for x in (dataset/'train.candidate.jsonl').read_text(encoding='utf-8').splitlines()]
    validation=[json.loads(x) for x in (dataset/'validation.candidate.jsonl').read_text(encoding='utf-8').splitlines()]
    assert entries and validation,'Training and validation data must be nonempty'
    if args.mode=='probe':
        ordered=sorted(entries,key=lambda r:r['duration'])
        entries=[ordered[0],ordered[len(ordered)//2],ordered[-1]]
        validation=validation[:1]
    emit('load_start',mode=args.mode,samples=len(entries))
    model=VoxCPM2Model.from_local(str(args.base_model.resolve()),training=True,
          optimize=False,device='cpu',lora_config=LoRAConfig(r=16,alpha=16,enable_lm=True,enable_dit=True),low_memory_training=True)
    model._publication_base_path=str(args.base_model.resolve())
    assert torch.get_default_dtype()==torch.float32
    for name,p in model.named_parameters():
        assert p.requires_grad==('lora_' in name),name
        assert p.dtype==(torch.float32 if p.requires_grad or name.startswith('audio_vae.') else torch.bfloat16),(name,p.dtype)
    # Encode once on GPU, then keep VAE off GPU during gradient computation.
    vae=model.audio_vae.eval()
    processor=BatchProcessor(config=model.config,audio_vae=vae,dataset_cnt=1,device=torch.device('cuda'))
    packed=[]
    for i,r in enumerate(entries+validation):
        a,sr=sf.read(dataset/r['audio'],dtype='float32')
        a=torchaudio.functional.resample(torch.from_numpy(a),sr,16000)
        sample={'text_ids':model.text_tokenizer(r['text']),'audio_array':a.numpy(),'audio_sampling_rate':16000,'dataset_id':0,'is_prompt':False}
        b=processor(HFVoxCPMDataset.collate_fn([sample]))
        packed.append({k:v.cpu() for k,v in b.items() if isinstance(v,torch.Tensor)})
        if i%25==0:emit('encoded',count=i+1,total=len(entries)+len(validation))
    del processor,b
    vae.cpu();del model.audio_vae
    gc.collect();torch.cuda.empty_cache()
    model.device='cuda';model.config.device='cuda'
    model.cuda().train()
    for module in model.modules():
        if isinstance(module,MiniCPMModel):module.gradient_checkpointing=True
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=.01,foreach=False)
    baseline_loss=evaluate(model,packed[len(entries):])
    emit('baseline_validation',loss=baseline_loss)
    initial={k:p.detach().cpu().clone() for k,p in model.named_parameters() if p.requires_grad}
    frozen_name,frozen=next((k,p) for k,p in model.named_parameters() if not p.requires_grad)
    frozen_slice=frozen.detach().flatten()[:256].cpu().clone()
    torch.cuda.reset_peak_memory_stats()
    steps=len(entries) if args.mode=='probe' else args.steps
    accum=1 if args.mode=='probe' else args.accum
    order=list(range(len(entries)));random.shuffle(order);cursor=0;metrics=[]
    start=time.monotonic()
    for step in range(steps):
        if args.mode=='train':
            import math
            factor=min((step+1)/4,1.0)*(.1+.9*(1+math.cos(math.pi*step/max(steps-1,1)))/2)
            for group in optimizer.param_groups:group['lr']=args.lr*factor
        optimizer.zero_grad(set_to_none=True)
        loss_value=0.0
        for micro in range(accum):
            if args.mode=='probe':idx=step
            else:
                if cursor==len(order):random.shuffle(order);cursor=0
                idx=order[cursor];cursor+=1
            result=forward(model,packed[idx]);loss=result['loss/diff']+result['loss/stop']
            assert torch.isfinite(loss).item(),'nonfinite loss'
            loss_value+=loss.item()/accum
            (loss/accum).backward()
            del result,loss
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True)
        assert float(norm)>0,'zero gradient'
        optimizer.step()
        torch.cuda.synchronize()
        metric=dict(step=step+1,loss=loss_value,grad_norm=float(norm),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,elapsed_seconds=time.monotonic()-start)
        metrics.append(metric);emit('step',**metric)
        if (step+1)%16==0 or step+1==steps:
            metric['validation_loss']=evaluate(model,packed[len(entries):])
            emit('validation',step=step+1,loss=metric['validation_loss'])
            save_adapter(model,out/f'step_{step+1:07}',step+1,optimizer)
    changed=[k for k,p in model.named_parameters() if p.requires_grad and not torch.equal(p.detach().cpu(),initial[k])]
    assert changed and any(k.startswith('base_lm.') for k in changed) and any(k.startswith('feat_decoder.') for k in changed)
    assert torch.equal(frozen.detach().flatten()[:256].cpu(),frozen_slice)
    validation_loss=evaluate(model,packed[len(entries):])
    # Exercise the model's real adapter loader, rather than just file deserialization.
    saved={k:p.detach().cpu().clone() for k,p in model.named_parameters() if p.requires_grad}
    model.reset_lora_weights()
    model.load_lora_weights(str(out/f'step_{steps:07}'))
    assert all(torch.equal(p.detach().cpu(),saved[k]) for k,p in model.named_parameters() if p.requires_grad)
    report=dict(mode=args.mode,steps=steps,grad_accum_steps=accum,train_samples=len(entries),validation_samples=len(validation),
                dataset=str(dataset),learning_rate=args.lr,
                baseline_validation_loss=baseline_loss,validation_loss=validation_loss,changed_lora_tensors=len(changed),frozen_slice_unchanged=True,
                adapter_save_reload_exact=True,metrics=metrics,training_data_review_status='machine_screened_unreviewed')
    (out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    emit('complete',**{k:v for k,v in report.items() if k!='metrics'})

if __name__=='__main__':main()
