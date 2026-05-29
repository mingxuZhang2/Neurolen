#!/bin/bash
#SBATCH --job-name=sr_imgtr
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/sr_imgtr_%j.log

# Extract LLaVA IMAGE features for the SemReps train IMAGE-TRIAL coco_ids
# (betas_train_image set, ~4897 ids) so the image->SEE encoder aligns to the
# image-viewing betas. Caption side was a different stimulus set.
# Output: results_semreps/semreps_train_imgtrial_features.npz

export PROJECT_DIR=/data/user/mzhang630/data/mllm
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python
export PYTHONUNBUFFERED=1
echo "Node: $(hostname)  GPU: $(nvidia-smi -L 2>/dev/null | head -1)  Date: $(date)"

$PY << 'PYEOF'
import os, json, logging, numpy as np, torch, gc
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('sr_imgtr')
from PIL import Image

PROJECT_DIR='/data/user/mzhang630/data/mllm'; SEMREPS_DIR='/data/user/mzhang630/data/semreps'
LLAVA_DIR=f'{PROJECT_DIR}/models/llava-v1.5-7b'
LAYERS=[0,4,8,12,14,16,20,24,31]; N_VIS=576

ids_all=[int(c) for c in json.load(open(f'{SEMREPS_DIR}/train_image_ids.json'))]
imgmap=json.load(open(f'{SEMREPS_DIR}/train_image_imgmap.json'))  # {str(id):'split/file.jpg'}
ids=[c for c in ids_all if str(c) in imgmap and os.path.exists(f'{SEMREPS_DIR}/images/{imgmap[str(c)]}')]
logger.info(f'{len(ids)}/{len(ids_all)} image-trial items with local image')

from transformers import CLIPVisionModel, CLIPImageProcessor, LlamaForCausalLM, LlamaConfig, AutoTokenizer
cfg=json.load(open(os.path.join(LLAVA_DIR,'config.json')))
clip_path=cfg.get('mm_vision_tower',''); dtype=torch.float16; device='cuda'
mm_hidden=cfg.get('mm_hidden_size',1024); hidden=cfg.get('hidden_size',4096)
vision_tower=CLIPVisionModel.from_pretrained(clip_path,torch_dtype=dtype).to(device).eval()
image_processor=CLIPImageProcessor.from_pretrained(clip_path)
projector=torch.nn.Sequential(torch.nn.Linear(mm_hidden,hidden),torch.nn.GELU(),torch.nn.Linear(hidden,hidden)).to(device,dtype=dtype)
idx=json.load(open(os.path.join(LLAVA_DIR,'pytorch_model.bin.index.json')))
proj_w={}; llm_state={}
for sf in sorted(set(idx['weight_map'].values())):
    shard=torch.load(os.path.join(LLAVA_DIR,sf),map_location='cpu',weights_only=True)
    for k,v in shard.items():
        ck=k
        for pre in ['model.language_model.','model.language_model.model.']:
            if ck.startswith(pre):
                ck=ck[len(pre):]
                if not ck.startswith('model.'): ck='model.'+ck
                break
        if 'mm_projector' in ck or 'multi_modal_projector' in ck: proj_w[ck]=v
        elif ck.startswith('model.vision_tower.'): pass
        elif ck.startswith('model.') or ck=='lm_head.weight': llm_state[ck]=v
    del shard
ps={}
for k,v in proj_w.items():
    c=k
    for pre in ('model.mm_projector.','model.multi_modal_projector.'):
        if c.startswith(pre): c=c[len(pre):]; break
    if c.startswith('linear_1'): c=c.replace('linear_1','0')
    if c.startswith('linear_2'): c=c.replace('linear_2','2')
    ps[c]=v
projector.load_state_dict(ps); projector.eval()
lc=LlamaConfig(vocab_size=cfg.get('vocab_size',32000),hidden_size=hidden,
    intermediate_size=cfg.get('intermediate_size',11008),num_hidden_layers=cfg.get('num_hidden_layers',32),
    num_attention_heads=cfg.get('num_attention_heads',32),num_key_value_heads=cfg.get('num_key_value_heads',32),
    max_position_embeddings=cfg.get('max_position_embeddings',4096),rms_norm_eps=cfg.get('rms_norm_eps',1e-5),torch_dtype=dtype)
model=LlamaForCausalLM(lc); model.load_state_dict(llm_state,strict=False); model=model.to(dtype=dtype).to(device).eval()
del llm_state, proj_w; gc.collect()
tokenizer=AutoTokenizer.from_pretrained(LLAVA_DIR,use_fast=False)
logger.info('Model loaded')

hooks=[]; acts={}
def mk(L):
    def f(m,i,o): acts[L]=(o[0] if isinstance(o,tuple) else o).detach()
    return f
for L in LAYERS: hooks.append(model.model.layers[L].register_forward_hook(mk(L)))

n=len(ids); feat={L:np.zeros((n,hidden),dtype=np.float32) for L in LAYERS}
prompt='USER: \nDescribe this image in detail.\nASSISTANT:'
logger.info('Extracting image-trial features...')
for i,cid in enumerate(ids):
    img=Image.open(f'{SEMREPS_DIR}/images/{imgmap[str(cid)]}').convert('RGB')
    pv=image_processor(images=img,return_tensors='pt')['pixel_values'].to(device,dtype=dtype)
    with torch.no_grad():
        vo=vision_tower(pv,output_hidden_states=True)
        patches=vo.hidden_states[-2][:,1:,:]; imfeat=projector(patches)
        tid=tokenizer(prompt,return_tensors='pt')['input_ids'].to(device)
        temb=model.get_input_embeddings()(tid)
        acts.clear(); model(inputs_embeds=torch.cat([imfeat,temb],dim=1))
        for L in LAYERS:
            h=acts[L][0] if acts[L].dim()==3 else acts[L]
            feat[L][i]=h[:N_VIS].cpu().float().mean(0).numpy()
    if (i+1)%200==0: logger.info(f'  {i+1}/{n}')
for h in hooks: h.remove()
from pathlib import Path
out=Path(f'{PROJECT_DIR}/results_semreps'); out.mkdir(parents=True,exist_ok=True)
np.savez(out/'semreps_train_imgtrial_features.npz', coco_ids=np.array(ids),
         **{f'layer_{L}':feat[L] for L in LAYERS})
logger.info(f'Saved image-trial features ({n} items)')
logger.info('Image-trial extraction complete')
PYEOF
echo "Done: $(date)"
