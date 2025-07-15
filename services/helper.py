from typing import List, Dict, Union, Tuple, Optional
import numpy as np
import cv2

def compute_iou(box1: Tuple[float, float, float, float], 
                box2: Tuple[float, float, float, float]) -> float:
    """Compute Intersection over Union for two bounding boxes [x, y, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection coordinates
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union > 0 else 0.0

def compute_center_error(box1: Tuple[float, float, float, float], 
                         box2: Tuple[float, float, float, float]) -> float:
    """Compute Euclidean distance between centers of two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    center1 = np.array([x1 + w1 / 2, y1 + h1 / 2])
    center2 = np.array([x2 + w2 / 2, y2 + h2 / 2])
    return np.linalg.norm(center1 - center2)

def validate_bbox(bbox: Union[List, Tuple], 
                  frame_shape: Tuple[int, int, int]) -> Tuple[float, float, float, float]:
    """Validate and convert bounding box to tuple of floats (x, y, w, h)."""
    try:
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x, y, w, h = map(float, bbox)
            if w > 0 and h > 0 and 0 <= x < frame_shape[1] and 0 <= y < frame_shape[0]:
                return (x, y, w, h)
        raise ValueError(f"Invalid bounding box format: {bbox}")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Bounding box validation failed: {e}")

def clamp_bbox_to_int(bbox: Tuple[float, float, float, float], 
                      img_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    """Convert bounding box to integers and clamp to image boundaries."""
    img_h, img_w = img_shape[:2]
    x, y, w, h = bbox
    
    # Convert to integers with rounding
    x = int(round(x))
    y = int(round(y))
    w = int(round(w))
    h = int(round(h))
    
    # Clamp coordinates to image dimensions
    x = max(0, min(img_w - 1, x))
    y = max(0, min(img_h - 1, y))
    w = max(1, min(img_w - x, w))
    h = max(1, min(img_h - y, h))
    
    return (x, y, w, h)

def draw_tracking_info(frame: np.ndarray, 
                       tracker_name: str, 
                       combo: str, 
                       frame_num: int, 
                       total_frames: int,
                       gt_bbox: Tuple[float, float, float, float],
                       pred_bbox: Optional[Tuple[float, float, float, float]],
                       iou: Optional[float],
                       failures: int) -> np.ndarray:
    """Draw tracking visualization on frame."""
    viz_frame = frame.copy()
    height, width = frame.shape[:2]
    
    # Draw ground truth (green)
    gt_rect = tuple(map(int, gt_bbox))
    cv2.rectangle(viz_frame, (gt_rect[0], gt_rect[1]), 
                (gt_rect[0]+gt_rect[2], gt_rect[1]+gt_rect[3]), 
                (0, 255, 0), 2)
    
    # Draw prediction if available (red)
    if pred_bbox is not None:
        pred_rect = tuple(map(int, pred_bbox))
        cv2.rectangle(viz_frame, (pred_rect[0], pred_rect[1]),
                    (pred_rect[0]+pred_rect[2], pred_rect[1]+pred_rect[3]),
                    (0, 0, 255), 2)
        
        # Draw center error if IoU available
        if iou is not None and iou > 0.1:
            gt_center = (int(gt_bbox[0] + gt_bbox[2]/2), int(gt_bbox[1] + gt_bbox[3]/2))
            pred_center = (int(pred_bbox[0] + pred_bbox[2]/2), int(pred_bbox[1] + pred_bbox[3]/2))
            cv2.line(viz_frame, gt_center, pred_center, (255, 0, 0), 2)
            cv2.circle(viz_frame, gt_center, 3, (0, 255, 255), -1)
            cv2.circle(viz_frame, pred_center, 3, (255, 255, 0), -1)
    
    # Add info text
    status = "Tracking" if pred_bbox is not None else "Failed"
    iou_text = f"IoU: {iou:.2f}" if iou is not None else "IoU: N/A"
    
    # Main info
    cv2.putText(viz_frame, f"{tracker_name} | {combo} | {status}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Frame info
    cv2.putText(viz_frame, f"Frame: {frame_num}/{total_frames}", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Metrics
    cv2.putText(viz_frame, f"{iou_text} | Failures: {failures}", 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Scale bar (10% of image width)
    scale_length = int(width * 0.1)
    cv2.line(viz_frame, (width-20, height-20), 
            (width-20-scale_length, height-20), (255, 255, 255), 2)
    cv2.putText(viz_frame, f"{scale_length}px", 
               (width-20-scale_length-50, height-15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return viz_frame