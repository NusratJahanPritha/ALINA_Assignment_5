from pathlib import Path
import argparse, csv, os, re, shutil, tempfile, time
import numpy as np
from alina.config import PipelineConfig
from alina.pipeline import run_batch

ROOT=Path('data/Raw_Data')
ROI_ROOT=Path('outputs/gmm_dynamic')
OUT_ROOT=Path('outputs/gmm_dynamic_alina')
VIDEOS=['vidd_1','vidd_2','vidd_3']


def to_alina_six(roi):
    bl,tl,tr,br=np.asarray(roi,dtype=np.float32).reshape(4,2)
    # alina.roi.roi_points_to_quad uses points 0, 1, 3.x, 5.x.
    # Its generated right-side y values come from TL and BL.
    return np.array([[bl,tl,tl,tr,tr,br]],dtype=np.float32)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--video',choices=VIDEOS,required=True)
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    frames=sorted((ROOT/args.video).glob('*.jpg'))
    if args.limit: frames=frames[:args.limit]
    if not frames: raise FileNotFoundError(ROOT/args.video)
    dest=OUT_ROOT/args.video
    for name in ('annotated','coords','birdseye'):
        (dest/name).mkdir(parents=True,exist_ok=True)
    rows=[]
    with tempfile.TemporaryDirectory(prefix='alina_dynamic_') as tmp:
        tmp=Path(tmp)
        for i,frame in enumerate(frames,1):
            roi_path=ROI_ROOT/args.video/'rois'/f'{frame.stem}.txt'
            if not roi_path.exists(): raise FileNotFoundError(roi_path)
            roi=np.loadtxt(roi_path,dtype=np.float32).reshape(4,2)
            one_input=tmp/'input';one_input.mkdir(exist_ok=True)
            for old in one_input.iterdir(): old.unlink()
            target=one_input/frame.name
            try: os.link(frame,target)
            except OSError: shutil.copy2(frame,target)
            one_out=tmp/'output'
            if one_out.exists(): shutil.rmtree(one_out)
            config=PipelineConfig(
                input_dir=one_input,
                output_images_dir=one_out/'annotated',
                output_coords_dir=one_out/'coords',
                log_file=one_out/'timing.log',
                peak_pixel_threshold=20,
                min_white_pixels=50,
                mask_ignore_left_columns=0,
            )
            start=time.perf_counter()
            # run_batch currently confirms the ROI interactively.
            # Confirm each frame when its ROI preview appears.
            run_batch(config,to_alina_six(roi))
            wall=time.perf_counter()-start
            log_path=one_out/'timing.log'
            log=log_path.read_text(encoding='utf-8',errors='replace') if log_path.exists() else ''
            matches=re.findall(r'\[\d+/\d+\]\s*\[([^\]]+)\].*?elapsed\s+([\d.]+)s',log)
            if len(matches)!=1:
                raise RuntimeError(f'Could not parse single-frame ALINA log for {frame.name}: {log[-1000:]}')
            status,elapsed=matches[0]
            for sub in ('annotated','coords','birdseye'):
                source=one_out/sub
                if source.exists():
                    for item in source.iterdir():
                        if item.is_file():
                            # Avoid collisions in birdseye/first_frame_warped.jpg.
                            name=(frame.stem+'_'+item.name) if sub=='birdseye' else item.name
                            shutil.copy2(item,dest/sub/name)
            rows.append(dict(frame=frame.name,status=status,alina_seconds=float(elapsed),run_wall_seconds=round(wall,6)))
            print(f'{args.video} [{i}/{len(frames)}] {frame.name}: {status}, ALINA {elapsed}s',flush=True)
            with (dest/'per_frame.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=['frame','status','alina_seconds','run_wall_seconds']);writer.writeheader();writer.writerows(rows)
    labeled=sum(r['status'].lower()=='labeled' for r in rows)
    print(f'Finished: {labeled}/{len(rows)} labeled ({100*labeled/len(rows):.1f}%)')

if __name__=='__main__': main()
