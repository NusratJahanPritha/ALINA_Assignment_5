from pathlib import Path
import argparse, csv, time
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

ROOT = Path('data/Raw_Data')
OUT = Path('outputs/gmm_dynamic')
VIDEOS = ['vidd_1', 'vidd_2', 'vidd_3']
MAX_DIM = 960
RNG = np.random.default_rng(42)


def estimate_roi(img):
    H, W = img.shape[:2]
    scale = min(1.0, MAX_DIM / max(H, W))
    w, h = round(W * scale), round(H * scale)
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) if scale < 1 else img
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64) / 255.0
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (10, 45, 70), (45, 255, 255)) > 0
    yellow[:int(.45*h)] = False
    pool = np.flatnonzero(np.indices((h,w))[0].ravel() >= int(.45*h))
    train_idx = RNG.choice(pool, size=min(15000,len(pool)), replace=False)
    gmm = GaussianMixture(n_components=5, covariance_type='diag', reg_covar=1e-3,
                          n_init=3, max_iter=100, random_state=42)
    gmm.fit(lab[train_idx])
    # Classify only yellow-reference pixels to reduce memory and runtime.
    yellow_ids = np.flatnonzero(yellow.ravel())
    if len(yellow_ids) < 30:
        raise ValueError('Fewer than 30 yellow-reference pixels')
    labels = gmm.predict(lab[yellow_ids])
    counts = np.bincount(labels, minlength=5)
    component = int(np.argmax(counts))
    selected = yellow_ids[labels == component]
    if len(selected) < 30:
        raise ValueError('Fewer than 30 selected yellow pixels')
    ys, xs = np.divmod(selected, w)
    xmin, xmax = np.percentile(xs, [5, 95]); ymin, ymax = np.percentile(ys, [5, 95])
    left = max(0., xmin-.12*w); right = min(w-1., xmax+.12*w)
    top = max(.45*h, ymin-.20*h); bottom = min(h-1., ymax+.20*h)
    if right-left < .12*w:
        cx = (left+right)/2; left=max(0.,cx-.06*w); right=min(w-1.,cx+.06*w)
    if bottom-top < .20*h:
        top=max(.45*h,bottom-.20*h); bottom=min(h-1.,max(bottom,top+.20*h))
    if right <= left or bottom <= top:
        raise ValueError('Invalid ROI')
    inset = .15*(right-left)
    roi = np.array([[left,bottom],[left+inset,top],[right-inset,top],[right,bottom]],dtype=np.float32)
    roi[:,0] *= W/w; roi[:,1] *= H/h
    mask = np.zeros((h,w),dtype=np.uint8)
    mask.ravel()[selected] = 255
    return roi, mask


def main():
    p=argparse.ArgumentParser(); p.add_argument('--video', choices=VIDEOS+['all'],default='all'); p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    for video in (VIDEOS if args.video=='all' else [args.video]):
        frames=sorted((ROOT/video).glob('*.jpg'))
        if args.limit: frames=frames[:args.limit]
        if not frames: raise FileNotFoundError(ROOT/video)
        dest=OUT/video; (dest/'rois').mkdir(parents=True,exist_ok=True); (dest/'previews').mkdir(exist_ok=True)
        rows=[]; previous=None
        for i,frame in enumerate(frames,1):
            t=time.perf_counter(); status='ok'; error=''
            try:
                image=cv2.imread(str(frame))
                if image is None: raise ValueError('Cannot read image')
                roi,mask=estimate_roi(image)
                previous=roi
            except Exception as e:
                status='fallback_previous'; error=str(e)
                if previous is None:
                    fallback=Path('outputs/gmm')/f'{video}_roi.txt'
                    if not fallback.exists(): raise RuntimeError(f'{frame.name}: {e}; missing fallback {fallback}') from e
                    roi=np.loadtxt(fallback,dtype=np.float32).reshape(4,2)
                    previous=roi
                else: roi=previous.copy()
            seconds=time.perf_counter()-t
            np.savetxt(dest/'rois'/f'{frame.stem}.txt',roi,fmt='%.2f')
            if i == 1 or i % 10 == 0:
                preview=cv2.imread(str(frame))
                if preview is not None:
                    cv2.polylines(preview,[np.round(roi).astype(np.int32)],True,(0,0,255),5)
                    cv2.imwrite(str(dest/'previews'/f'{frame.stem}.jpg'),preview)
            rows.append(dict(frame=frame.name,roi_seconds=round(seconds,6),status=status,error=error))
            print(f'{video} [{i}/{len(frames)}] {frame.name}: {status}, ROI {seconds:.2f}s',flush=True)
        with (dest/'roi_timing.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['frame','roi_seconds','status','error']);writer.writeheader();writer.writerows(rows)
        print(f'Saved {len(rows)} ROIs in {dest}')

if __name__=='__main__': main()
