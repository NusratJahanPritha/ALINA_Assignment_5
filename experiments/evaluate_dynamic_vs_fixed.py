from pathlib import Path
import csv, re
import numpy as np
import cv2

# VIDEOS=['vidd_1','vidd_2','vidd_3']
VIDEOS=['vidd_1']

FIXED=Path('outputs/gmm_alina')
DYNAMIC=Path('outputs/gmm_dynamic_alina')
ROI=Path('outputs/gmm_dynamic')
OUT=Path('outputs/gmm_dynamic/comparison.csv')


def fixed_stats(video):
    path=FIXED/video/'timing.log'
    if not path.exists(): return None
    log=path.read_text(encoding='utf-8',errors='replace')
    matches=re.findall(r'\[\d+/\d+\]\s*\[([^\]]+)\].*?elapsed\s+([\d.]+)s',log)
    if not matches: return None
    return { 'labeled':sum(s.lower()=='labeled' for s,t in matches),
             'total':len(matches),'alina_ms':1000*np.mean([float(t) for s,t in matches]) }


def dynamic_stats(video):
    path=DYNAMIC/video/'per_frame.csv'
    if not path.exists(): return None
    with path.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    if not rows: return None
    return {'labeled':sum(r['status'].lower()=='labeled' for r in rows),
            'total':len(rows),'alina_ms':1000*np.mean([float(r['alina_seconds']) for r in rows])}


def roi_times(video):
    path=ROI/video/'roi_timing.csv'
    if not path.exists(): return {}
    with path.open(newline='',encoding='utf-8') as f:
        return {r['frame']:float(r['roi_seconds']) for r in csv.DictReader(f)}


def roi_iou(a,b,shape):
    h,w=shape
    x=np.zeros((h,w),dtype=np.uint8);y=np.zeros_like(x)
    cv2.fillConvexPoly(x,np.round(a).astype(np.int32),1)
    cv2.fillConvexPoly(y,np.round(b).astype(np.int32),1)
    union=np.count_nonzero(x|y)
    return float(np.count_nonzero(x&y)/union) if union else float('nan')


def mean_roi_iou(video):
    folder=ROI/video/'rois'
    files=sorted(folder.glob('*.txt'))
    if len(files)<2:return None
    first=cv2.imread(str(sorted((Path('data/Raw_Data')/video).glob('*.jpg'))[0]))
    if first is None:return None
    shape=first.shape[:2]
    values=[]; prev=np.loadtxt(files[0]).reshape(4,2)
    for file in files[1:]:
        cur=np.loadtxt(file).reshape(4,2)
        values.append(roi_iou(prev,cur,shape));prev=cur
    return float(np.mean(values))


def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    records=[]
    for video in VIDEOS:
        fixed=fixed_stats(video);dyn=dynamic_stats(video)
        if fixed is None or dyn is None:
            print(f'{video}: missing fixed or dynamic results');continue
        if fixed['total']!=dyn['total']:
            print(f'WARNING {video}: fixed {fixed["total"]} frames, dynamic {dyn["total"]}; compare matching frame sets only')
            continue
        rt=roi_times(video)
        frames=sorted((Path('data/Raw_Data')/video).glob('*.jpg'))[:dyn['total']]
        if any(f.name not in rt for f in frames):
            print(f'{video}: missing ROI times for some frames');continue
        roi_ms=1000*np.mean([rt[f.name] for f in frames])
        fixed_roi_ms=1000*rt[frames[0].name]/len(frames) # illustrative only if same first-frame ROI algorithm
        row={
            'video':video,'frames':fixed['total'],
            'fixed_labeled':fixed['labeled'],'dynamic_labeled':dyn['labeled'],
            'fixed_label_rate_pct':round(100*fixed['labeled']/fixed['total'],2),
            'dynamic_label_rate_pct':round(100*dyn['labeled']/dyn['total'],2),
            'label_rate_difference_pp':round(100*(dyn['labeled']-fixed['labeled'])/fixed['total'],2),
            'fixed_alina_ms_per_frame':round(fixed['alina_ms'],2),
            'dynamic_alina_ms_per_frame':round(dyn['alina_ms'],2),
            'dynamic_roi_ms_per_frame':round(roi_ms,2),
            'dynamic_total_ms_per_frame':round(dyn['alina_ms']+roi_ms,2),
            'mean_consecutive_roi_iou':round(mean_roi_iou(video),4),
        }
        records.append(row)
        print(row)
    if records:
        with OUT.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        print('Saved:',OUT)

if __name__=='__main__':main()
