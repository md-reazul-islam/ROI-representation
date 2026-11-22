from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from .base import BaseDetector, Detection


@dataclass(frozen=True)
class RTDETRDetectionRecord:
    video_id: str
    split: str
    frame_index: int
    timestamp_sec: float
    detection_status: str
    ground_truth_status: str
    class_id: int | None
    class_name: str | None
    confidence: float | None
    x1: float | None
    y1: float | None
    x2: float | None
    y2: float | None
    bbox_width: float | None
    bbox_height: float | None
    selected_detection: bool
    num_candidates: int
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()


class RTDETRDetector(BaseDetector):
    """Independent RT-DETR adapter. Does not share code or state with YOLODetector."""

    def __init__(self, model_path: str | Path, confidence: float = 0.25, iou: float = 0.45,
                 imgsz: int = 640, device: str = "auto", class_id: int = 0):
        self.model_path=Path(model_path); self.confidence=float(confidence); self.iou=float(iou)
        self.imgsz=int(imgsz); self.device=device; self.class_id=int(class_id); self._model=None

    def load(self):
        try: from ultralytics import RTDETR
        except ImportError as exc:
            raise RuntimeError("Ultralytics is required for Phase 3. Install with `python -m pip install ultralytics`.") from exc
        if not self.model_path.exists(): raise FileNotFoundError(f"RT-DETR model checkpoint not found: {self.model_path}")
        self._model=RTDETR(str(self.model_path))

    @property
    def model(self):
        if self._model is None: self.load()
        return self._model

    def predict(self, frame: np.ndarray):
        kwargs=dict(conf=self.confidence,iou=self.iou,imgsz=self.imgsz,classes=[self.class_id],verbose=False)
        if self.device and self.device!="auto": kwargs["device"]=self.device
        return self.model.predict(frame,**kwargs)[0]

    def extract_candidates(self,result):
        boxes=getattr(result,"boxes",None)
        if boxes is None or len(boxes)==0: return []
        xyxy=boxes.xyxy.detach().cpu().numpy(); conf=boxes.conf.detach().cpu().numpy(); cls=boxes.cls.detach().cpu().numpy().astype(int)
        names=getattr(result,"names",{})
        out=[]
        for i in range(len(xyxy)):
            cid=int(cls[i]); cname=names.get(cid,str(cid)) if isinstance(names,dict) else str(cid)
            x1,y1,x2,y2=map(float,xyxy[i])
            out.append({"class_id":cid,"class_name":cname,"confidence":float(conf[i]),"x1":x1,"y1":y1,"x2":x2,"y2":y2,
                        "bbox_width":max(0.0,x2-x1),"bbox_height":max(0.0,y2-y1)})
        return sorted(out,key=lambda x:x["confidence"],reverse=True)

    def detect(self, frame: np.ndarray, frame_index: int = 0, timestamp: float = 0.0) -> list[Detection]:
        result=self.predict(frame)
        candidates=self.extract_candidates(result)
        return [Detection(bbox=(c["x1"],c["y1"],c["x2"],c["y2"]),confidence=c["confidence"],
                           class_id=c["class_id"],class_name=c["class_name"],
                           frame_index=frame_index,timestamp=timestamp) for c in candidates]
