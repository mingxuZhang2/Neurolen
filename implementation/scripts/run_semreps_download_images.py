"""Download COCO stimulus images for SemReps subjects from images.cocodataset.org.

Reads a {coco_id: "split/filename.jpg"} imgmap (relative under images/), skips
files already present, and fetches the rest with a thread pool + retries.
HPC2 has outbound internet; HPC3 does not -> download here, rsync to HPC3.

Usage:
  python run_semreps_download_images.py <imgmap.json> [images_root]
    imgmap.json : {"529": "train2017/000000000529.jpg", ...}
    images_root : default /hpc2hdd/home/mzhang630/data/semreps/images
"""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

IMGMAP = sys.argv[1]
ROOT = sys.argv[2] if len(sys.argv) > 2 else '/hpc2hdd/home/mzhang630/data/semreps/images'
BASE = 'http://images.cocodataset.org'
N_THREADS, RETRIES = 16, 4

imgmap = json.load(open(IMGMAP))
todo = [(cid, rel) for cid, rel in imgmap.items() if not os.path.exists(os.path.join(ROOT, rel))]
print(f'{len(imgmap)} total, {len(todo)} to download into {ROOT}', flush=True)
for split in {rel.split('/')[0] for rel in imgmap.values()}:
    os.makedirs(os.path.join(ROOT, split), exist_ok=True)


def fetch(item):
    cid, rel = item
    url = f'{BASE}/{rel}'
    dst = os.path.join(ROOT, rel)
    for a in range(RETRIES):
        try:
            urllib.request.urlretrieve(url, dst + '.part')
            os.replace(dst + '.part', dst)
            return (cid, True, '')
        except Exception as e:
            if a == RETRIES - 1:
                return (cid, False, str(e))
            time.sleep(1.5 * (a + 1))


ok = fail = 0
fails = []
t0 = time.time()
with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
    futs = [ex.submit(fetch, it) for it in todo]
    for i, f in enumerate(as_completed(futs)):
        cid, good, err = f.result()
        if good:
            ok += 1
        else:
            fail += 1
            fails.append((cid, err))
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(todo)}  ok={ok} fail={fail}  {time.time()-t0:.0f}s', flush=True)

print(f'DONE ok={ok} fail={fail}  {time.time()-t0:.0f}s', flush=True)
if fails:
    json.dump(fails, open(IMGMAP.replace('.json', '_failures.json'), 'w'))
    print('first failures:', fails[:5], flush=True)
