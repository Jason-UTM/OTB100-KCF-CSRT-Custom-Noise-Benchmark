import os
import time
import cv2
import numpy as np
from tqdm import tqdm
from typing import Union, List, Dict, Optional, Tuple
from services.data import load_otb100_data
from services.helper import compute_iou, compute_center_error, validate_bbox, clamp_bbox_to_int, draw_tracking_info
from services.results import save_tracker_results, save_consolidated_results

def run_trackers_and_evaluate(
    dataset_path: str,
    results_dir: str,
    sequences: Union[str, List[str], None] = None,
    trackers: Optional[List[str]] = None,
    visualize: bool = True,
    save_videos: bool = False,
    failure_threshold: float = 0.1,
    frame_delay: int = 25
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Run specified trackers on noisy frames for OTB-100 sequences, evaluating metrics including time and failure counts.
    Saves results as CSVs and videos in results_dir/sequence_name.
    Prints total evaluation time to console.
    
    Args:
        dataset_path: Path to OTB-100 dataset
        results_dir: Directory to store results and videos (e.g., 'results')
        sequences: Sequence name(s) to process (None for all)
        trackers: List of tracker names (e.g., ['KCF', 'CSRT']) or None for defaults
        visualize: Show real-time tracking visualization
        save_videos: Save tracking results as videos
        failure_threshold: IoU threshold to consider tracking failed
        frame_delay: Delay between frames in ms (for visualization)
        
    Returns:
        Dictionary containing tracking results with metrics:
        {
            sequence_name: {
                tracker_name: {
                    noise_occlusion_combo: {
                        'EAO': float,
                        'Robustness': float,
                        'Precision': float,
                        'TrackingTime': float,
                        'FPS': float,
                        'NumFrames': int,
                        'NumFailures': int
                    }
                }
            }
        }
    """
    # Record start time for total evaluation
    total_start_time = time.time()
    
    # Define available trackers
    available_trackers = {
        'KCF': cv2.TrackerKCF_create,
        'CSRT': cv2.TrackerCSRT_create
    }
    
    # Check OpenCV tracker availability
    required_trackers = ['TrackerKCF_create', 'TrackerCSRT_create']
    missing_trackers = [t for t in required_trackers if not hasattr(cv2, t)]
    if missing_trackers:
        raise ImportError(
            f"OpenCV tracking module missing functions: {', '.join(missing_trackers)}. "
            "Install with: pip uninstall opencv-python opencv-contrib-python && "
            "pip install opencv-contrib-python==4.10.0.84"
        )
    
    # Validate trackers from config
    if trackers is None:
        trackers = list(available_trackers.keys())
    else:
        invalid_trackers = [t for t in trackers if t not in available_trackers]
        if invalid_trackers:
            raise ValueError(f"Invalid trackers specified: {invalid_trackers}. Available: {list(available_trackers.keys())}")
    
    # Create tracker dictionary based on config
    trackers_dict = {t: available_trackers[t] for t in trackers}
    
    # Normalize sequences input
    if sequences is None:
        sequence_list = None
    elif isinstance(sequences, str):
        sequence_list = [sequences]
    else:
        sequence_list = sequences
    
    # Load sequence data
    data = load_otb100_data(dataset_path, sequences=sequence_list)
    if not data:
        raise ValueError(f"No valid sequences found in {dataset_path}")
    
    # Define noise/occlusion combinations
    NOISE_TYPES = ['gaussian', 'salt_pepper']
    OCCLUSION_LEVELS = [0.2, 0.4, 0.6]
    combinations = [f"{nt}_{ol}" for nt in NOISE_TYPES for ol in OCCLUSION_LEVELS]
    
    # Prepare results structure
    results = {}
    
    # Initialize progress bar
    total_tasks = len(data) * len(trackers_dict) * len(combinations)
    with tqdm(total=total_tasks, desc="Tracking and evaluating", unit="task") as pbar:
        for seq, seq_data in data.items():
            results[seq] = {}
            gt_bboxes = seq_data['groundtruth_bboxes']
            initial_bbox = seq_data['initial_bbox']
            
            for tracker_name, tracker_create in trackers_dict.items():
                results[seq][tracker_name] = {}
                
                for combo in combinations:
                    # Initialize metrics
                    overlaps = []
                    center_errors = []
                    failures = 0
                    start_time = time.time()
                    
                    # Get frame paths
                    frame_dir = os.path.join(dataset_path, seq, f"img_{combo}")
                    if not os.path.exists(frame_dir):
                        print(f"Warning: Missing frames for {seq}/{combo}")
                        pbar.update(1)
                        continue
                    
                    frames = sorted([
                        os.path.join(frame_dir, f) 
                        for f in os.listdir(frame_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                    ])
                    
                    if not frames or len(frames) != len(gt_bboxes):
                        print(f"Warning: Frame count mismatch in {seq}/{combo}")
                        pbar.update(1)
                        continue
                    
                    # Initialize video writer
                    video_writer = None
                    if save_videos:
                        seq_dir = os.path.join(results_dir, seq)
                        os.makedirs(seq_dir, exist_ok=True)
                        video_path = os.path.join(seq_dir, f"{tracker_name.lower()}_{combo}.avi")
                        
                        # Check if video exists and notify about overwriting
                        if os.path.exists(video_path):
                            print(f"Overwriting existing video: {video_path}")
                        else:
                            print(f"Creating new video: {video_path}")
                        
                        first_frame = cv2.imread(frames[0])
                        if first_frame is not None:
                            height, width = first_frame.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'XVID')
                            video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
                    
                    # Initialize tracker with first frame
                    first_frame = cv2.imread(frames[0])
                    if first_frame is None:
                        print(f"Warning: Failed to load first frame for {seq}")
                        pbar.update(1)
                        continue
                    
                    try:
                        validated_bbox = validate_bbox(initial_bbox, first_frame.shape)
                        bbox_int = clamp_bbox_to_int(validated_bbox, first_frame.shape)
                        tracker = tracker_create()
                        tracker.init(first_frame, bbox_int)
                        
                        # Draw initial frame
                        if visualize or video_writer:
                            viz_frame = draw_tracking_info(
                                first_frame, tracker_name, combo, 1, len(frames),
                                validated_bbox, validated_bbox, 1.0, 0
                            )
                            
                            if visualize:
                                cv2.imshow('Tracking', viz_frame)
                                cv2.waitKey(1)
                            
                            if video_writer:
                                video_writer.write(viz_frame)
                    except Exception as e:
                        print(f"Tracker init failed for {seq}/{tracker_name}: {e}")
                        pbar.update(1)
                        continue
                    
                    # Process subsequent frames
                    for i in range(1, len(frames)):
                        frame_path = frames[i]
                        frame = cv2.imread(frame_path)
                        if frame is None:
                            print(f"Warning: Failed to load frame {i+1} for {seq}")
                            overlaps.append(0.0)
                            failures += 1
                            continue
                        
                        # Get ground truth
                        try:
                            gt_bbox = validate_bbox(gt_bboxes[i], frame.shape)
                        except ValueError as e:
                            print(f"Invalid GT bbox at frame {i+1} for {seq}: {e}")
                            gt_bbox = (0, 0, 10, 10)  # Fallback bbox
                        
                        # Update tracker
                        success, pred_bbox = tracker.update(frame)
                        pred_bbox = tuple(float(x) for x in pred_bbox) if success else None
                        
                        # Calculate metrics
                        iou = compute_iou(pred_bbox, gt_bbox) if success else 0.0
                        overlaps.append(iou)
                        
                        if not success or iou < failure_threshold:
                            failures += 1
                            try:
                                reinit_bbox = clamp_bbox_to_int(gt_bbox, frame.shape)
                                tracker = tracker_create()
                                tracker.init(frame, reinit_bbox)
                                pred_bbox = None  # Mark as failed for visualization
                            except Exception as e:
                                print(f"Re-init failed at frame {i+1}: {e}")
                        elif success:
                            center_errors.append(compute_center_error(pred_bbox, gt_bbox))
                        
                        # Visualization
                        if visualize or video_writer:
                            viz_frame = draw_tracking_info(
                                frame, tracker_name, combo, i+1, len(frames),
                                gt_bbox, pred_bbox, iou if success else None, failures
                            )
                            
                            if visualize:
                                cv2.imshow('Tracking', viz_frame)
                                key = cv2.waitKey(frame_delay) & 0xFF
                                if key == ord('q'):
                                    visualize = False
                            
                            if video_writer:
                                video_writer.write(viz_frame)
                    
                    # Calculate final metrics
                    end_time = time.time()
                    tracking_time = end_time - start_time
                    fps = len(frames) / tracking_time if tracking_time > 0 else 0.0
                    
                    successful_overlaps = [o for o in overlaps if o > 0]
                    eao = np.mean(successful_overlaps) if successful_overlaps else 0.0
                    robustness = failures / len(frames) if frames else 1.0
                    precision = np.mean(center_errors) if center_errors else float('inf')
                    
                    results[seq][tracker_name][combo] = {
                        'EAO': eao,
                        'Robustness': robustness,
                        'Precision': precision,
                        'TrackingTime': tracking_time,
                        'FPS': fps,
                        'NumFrames': len(frames),
                        'NumFailures': failures
                    }
                    
                    # Clean up video writer
                    if video_writer:
                        video_writer.release()
                    
                    pbar.update(1)
    
    # Save results to consolidated CSVs
    save_consolidated_results(results, results_dir, sequences=sequence_list)
    
    # Clean up visualization
    if visualize:
        cv2.destroyAllWindows()
    
    # Print total evaluation time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"Total evaluation time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    
    return results