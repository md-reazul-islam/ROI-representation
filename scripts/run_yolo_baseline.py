from __future__ import annotations
import argparse, json, platform
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2, pandas as pd
from module1.config import load_config, project_path
from module1.data.manifest import DatasetManifest
from module1.data.sampler import FrameSampler
from module1.data.video import VideoReader
from module1.detection.yolo_detector import YOLODetector, DetectionRecord
from module1.detection.metrics import calculate_detection_metrics, aggregate_metrics
from module1.logging_utils import setup_logging

def norm_device(v):
    if not v or v == 'auto':
        try:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception: return 'cpu'
    return v

def safe_name(s): return ''.join(c if c.isalnum() or c in '._-' else '_' for c in str(s))

def frame_path(root, video_id, frame_index): return root / f'{safe_name(video_id)}_frame_{frame_index:08d}.jpg'

def annotate(frame, record):
    canvas=frame.copy()
    if record.detection_status=='DETECTED':
        x1,y1,x2,y2=map(int,[record.x1,record.y1,record.x2,record.y2])
        cv2.rectangle(canvas,(x1,y1),(x2,y2),(255,255,255),2)
        cv2.putText(canvas,f'{record.class_name} conf={record.confidence:.3f}',(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(canvas,f'{record.video_id} | frame={record.frame_index} | t={record.timestamp_sec:.3f}s | {record.detection_status}',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2,cv2.LINE_AA)
    return canvas

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',default='dataset_manifest.csv'); p.add_argument('--config',default='configs/module1.yaml')
    p.add_argument('--device'); p.add_argument('--conf',type=float); p.add_argument('--imgsz',type=int); p.add_argument('--iou',type=float)
    args=p.parse_args()
    cfg=load_config(args.config); root=Path(cfg['_project_root']); logger=setup_logging(project_path(cfg, cfg['logging']['log_dir']), cfg['logging']['level'])
    mp=Path(args.manifest); mp=mp if mp.is_absolute() else root/mp
    manifest=DatasetManifest.load(mp); records=manifest.to_records()
    dcfg=cfg.get('detection',{}).get('yolo',{}); proc_fps=float(cfg.get('processing',{}).get('target_fps',20.0))
    device=norm_device(args.device or dcfg.get('device',cfg.get('experiment',{}).get('device','auto')))
    detector=YOLODetector(project_path(cfg,dcfg.get('model_path','models/yolo11s.pt')),
        confidence=args.conf if args.conf is not None else float(dcfg.get('confidence_threshold',.25)),
        iou=args.iou if args.iou is not None else float(dcfg.get('iou_threshold',.45)),
        imgsz=args.imgsz if args.imgsz is not None else int(dcfg.get('image_size',640)),device=device,class_id=int(dcfg.get('class_id',0)))
    out=project_path(cfg,'outputs/yolo'); vis=out/'visualizations'; metaout=out/'metadata'; vis.mkdir(parents=True,exist_ok=True); metaout.mkdir(parents=True,exist_ok=True)
    all_rows=[]; per_video=[]; native_meta=[]
    logger.info('Phase 2 starting: %d videos',len(records))
    for rec in records:
        row=manifest.df[manifest.df.video_id==rec.video_id].iloc[0]; vp=manifest.resolve_video_path(root,row)
        if not vp.exists(): raise FileNotFoundError(f'{rec.video_id}: {vp}')
        reader=VideoReader(vp); vm=reader.metadata(); native_meta.append({'video_id':rec.video_id,'split':rec.split,**vm.__dict__})
        rows=[]; sampler=FrameSampler(reader,proc_fps)
        for sampled in sampler.iter_sampled():
            result=detector.predict(sampled.frame); candidates=detector.extract_candidates(result); n=len(candidates)
            if not candidates:
                dr=DetectionRecord(rec.video_id,rec.split,sampled.frame_index,sampled.timestamp_sec,'MISSED','UNAVAILABLE',None,None,None,None,None,None,None,None,None,False,0); rows.append(dr.to_dict())
                if dcfg.get('save_annotated_frames',True): cv2.imwrite(str(frame_path(vis,rec.video_id,sampled.frame_index)),annotate(sampled.frame,dr))
            else:
                for i,c in enumerate(candidates):
                    dr=DetectionRecord(rec.video_id,rec.split,sampled.frame_index,sampled.timestamp_sec,'DETECTED','UNAVAILABLE',c['class_id'],c['class_name'],c['confidence'],c['x1'],c['y1'],c['x2'],c['y2'],c['bbox_width'],c['bbox_height'],i==0,n); rows.append(dr.to_dict())
                if dcfg.get('save_annotated_frames',True):
                    dr=DetectionRecord(**rows[-n]); cv2.imwrite(str(frame_path(vis,rec.video_id,sampled.frame_index)),annotate(sampled.frame,dr))
        all_rows.extend(rows); m=calculate_detection_metrics(rows); m.update({'video_id':rec.video_id,'split':rec.split,'native_fps':vm.native_fps,'width':vm.width,'height':vm.height,'frame_count_native':vm.frame_count,'duration_sec_native':vm.duration_sec}); per_video.append(m)
        logger.info('%s: %.4f detection rate; %d gaps; longest miss=%d',rec.video_id,m['detection_rate'],m['number_of_detection_gaps'],m['longest_consecutive_miss'])
    out.mkdir(parents=True,exist_ok=True); pd.DataFrame(all_rows).to_csv(out/'yolo_detections.csv',index=False)
    metric_rows=per_video+aggregate_metrics(all_rows,'split')+aggregate_metrics(all_rows,None); pd.DataFrame(metric_rows).to_csv(out/'yolo_metrics.csv',index=False)
    try:
        import ultralytics; yv=ultralytics.__version__
    except Exception: yv='unknown'
    metadata={'phase':'PHASE_2_YOLO_BASELINE','timestamp_utc':datetime.now(timezone.utc).isoformat(),'dataset_source':cfg.get('dataset',{}).get('source_name'),
      'manifest_path':str(mp.relative_to(root)) if mp.is_relative_to(root) else str(mp),'model_path':str(detector.model_path.relative_to(root)) if detector.model_path.is_relative_to(root) else str(detector.model_path),
      'model_name':'YOLO','model_version':yv,'class_id':detector.class_id,'candidate_name':dcfg.get('candidate_name','infant_candidate'),'confidence_threshold':detector.confidence,'iou_threshold':detector.iou,'image_size':detector.imgsz,
      'processing_fps':proc_fps,'native_fps_preserved':True,'device':device,'split_policy':{'train':'videos 1-18','validation':'videos 19-21','test':'videos 22-24','unit':'video'},
      'ground_truth_status':'UNAVAILABLE','metric_interpretation':'detection_rate_and_miss_rate_are_detector_observation_metrics_not_ground_truth_accuracy','python_version':platform.python_version(),'opencv_version':cv2.__version__,'native_video_metadata':native_meta}
    (metaout/'experiment_metadata.json').write_text(json.dumps(metadata,indent=2,default=str),encoding='utf-8')
    return 0
if __name__=='__main__': raise SystemExit(main())
