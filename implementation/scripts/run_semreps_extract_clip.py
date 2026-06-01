"""CLIP baseline feature extraction for the matching-transfer control.

LLaVA = CLIP-ViT-L/14-336 -> projector -> LLaMA. This extracts the CLIP image/text
embeddings in the SHARED projected space (get_image_features / get_text_features,
768-d, contrastively aligned) so we can run the IDENTICAL cross-modal transfer
analysis with the LLaMA decoder removed. If CLIP shows the same amodal-cortex
transfer gradient, the effect is generic to vision-language alignment; if weaker,
the decoder is contributing the brain-relevant amodal code.

Output (npz, drop-in for run_semreps_matching with SR_LAYERS=0):
  clip_image_features.npz       coco_ids + layer_0 (n_img, 768)   all train+test image ids
  clip_caption_features.npz     coco_ids + layer_0 (n_cap, 768)   all train+test caption ids
  clip_test_image_features.npz  coco_ids + layer_0 (70, 768)      test, ordered as semreps test
  clip_test_caption_features.npz coco_ids + layer_0 (70, 768)

Runs on an HPC2 GPU node.
"""
import json, glob, logging
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('clip_extract')

SR = Path('/hpc2hdd/home/mzhang630/data/semreps')
# local copy (same CLIP tower LLaVA uses); GPU node has no outbound internet
MODEL = '/hpc2hdd/home/mzhang630/data/mllm/transfer/models/clip-vit-large-patch14-336'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_BATCH = 64

IMG_SRC = ['semreps_train_imgtrial_features.npz', 'semreps_s23_image_features.npz',
           'semreps_s457_image_features.npz']
CAP_SRC = ['semreps_train_caption_features.npz', 'semreps_s23_caption_features.npz',
           'semreps_s457_caption_features.npz']
CAP_TXT = ['train_captions.json', 's23_new_caption_text.json', 's457_new_caption_text.json',
           'test_captions.json']  # test captions live in their own file


def union_ids(srcs):
    ids = []
    seen = set()
    for f in srcs:
        z = np.load(SR / f)
        for c in z['coco_ids']:
            c = int(c)
            if c not in seen:
                seen.add(c); ids.append(c)
    return ids


def img_path(cid):
    hits = glob.glob(str(SR / f'images/*/{cid:012d}.jpg'))
    return hits[0] if hits else None


def main():
    logger.info(f'device={DEVICE} model={MODEL}')
    model = CLIPModel.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEVICE).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)

    test_ids = [int(c) for c in np.load(SR / 'semreps_test_features.npz')['coco_ids']]
    train_img_ids = union_ids(IMG_SRC)
    train_cap_ids = union_ids(CAP_SRC)
    all_img_ids = train_img_ids + [c for c in test_ids if c not in set(train_img_ids)]
    all_cap_ids = train_cap_ids + [c for c in test_ids if c not in set(train_cap_ids)]
    cap_txt = {}
    for f in CAP_TXT:
        for k, v in json.load(open(SR / f)).items():
            cap_txt.setdefault(int(k), v)
    logger.info(f'img ids={len(all_img_ids)} cap ids={len(all_cap_ids)} test={len(test_ids)}')

    # ---- image embeddings ----
    @torch.no_grad()
    def embed_images(ids):
        out, kept = [], []
        batch_imgs, batch_ids = [], []
        def flush():
            if not batch_imgs:
                return
            px = proc(images=batch_imgs, return_tensors='pt').to(DEVICE)
            px = {k: v.half() if v.dtype == torch.float32 else v for k, v in px.items()}
            feat = model.get_image_features(**px).float().cpu().numpy()
            out.append(feat); kept.extend(batch_ids)
            batch_imgs.clear(); batch_ids.clear()
        for i, cid in enumerate(ids):
            p = img_path(cid)
            if p is None:
                continue
            try:
                batch_imgs.append(Image.open(p).convert('RGB')); batch_ids.append(cid)
            except Exception as e:
                logger.warning(f'img {cid} load fail: {e}'); continue
            if len(batch_imgs) >= IMG_BATCH:
                flush()
            if (i + 1) % 2560 == 0:
                logger.info(f'  images {i+1}/{len(ids)}')
        flush()
        return np.array(kept), np.concatenate(out).astype(np.float32)

    # ---- text embeddings ----
    @torch.no_grad()
    def embed_texts(ids):
        out, kept = [], []
        for s in range(0, len(ids), 256):
            chunk = ids[s:s + 256]
            ck = [c for c in chunk if c in cap_txt]
            txts = [cap_txt[c] for c in ck]
            if not txts:
                continue
            tok = proc(text=txts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            feat = model.get_text_features(**tok).float().cpu().numpy()
            out.append(feat); kept.extend(ck)
            if (s + 256) % 2560 == 0:
                logger.info(f'  texts {s+256}/{len(ids)}')
        return np.array(kept), np.concatenate(out).astype(np.float32)

    logger.info('extracting image embeddings...')
    img_ids, img_feat = embed_images(all_img_ids)
    logger.info(f'  image embeds: {img_feat.shape}')
    logger.info('extracting text embeddings...')
    cap_ids, cap_feat = embed_texts(all_cap_ids)
    logger.info(f'  text embeds: {cap_feat.shape}')

    np.savez(SR / 'clip_image_features.npz', coco_ids=img_ids, layer_0=img_feat)
    np.savez(SR / 'clip_caption_features.npz', coco_ids=cap_ids, layer_0=cap_feat)

    # test files, ordered exactly as semreps test set
    img_row = {int(c): i for i, c in enumerate(img_ids)}
    cap_row = {int(c): i for i, c in enumerate(cap_ids)}
    assert all(c in img_row and c in cap_row for c in test_ids), 'test id missing an embedding'
    te_img = img_feat[[img_row[c] for c in test_ids]]
    te_cap = cap_feat[[cap_row[c] for c in test_ids]]
    np.savez(SR / 'clip_test_image_features.npz', coco_ids=np.array(test_ids), layer_0=te_img)
    np.savez(SR / 'clip_test_caption_features.npz', coco_ids=np.array(test_ids), layer_0=te_cap)
    logger.info('Saved clip_{image,caption,test_image,test_caption}_features.npz')


if __name__ == '__main__':
    main()
