import os
import time
import cv2
import numpy as np
from tqdm import tqdm
from typing import Union, List, Dict, Optional, Tuple
from services.data import load_otb100_data
from services.helper import compute_iou, compute_center_error, validate_bbox, clamp_bbox_to_int, draw_tracking_info
from services.results import save_tracker_results, save_consolidated_results, save_all_sequences_consolidated_csv
from services.analysis import plot_metrics, plot_eao_trends, plot_precision_vs_robustness, generate_metrics_table

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
            # Check if sequence already has complete results (CSV + videos + plots)
            seq_dir = os.path.join(results_dir, seq)
            csv_path = os.path.join(seq_dir, f"{seq}_metrics_table.csv")
            
            # Function to check if sequence is completely processed
            def analyze_sequence_completion(seq_name, seq_directory, trackers, combinations, save_videos_flag):
                """Analyze what components are missing for a sequence."""
                missing_components = []
                existing_combinations = []
                
                # Check CSV file and extract existing combinations
                csv_file = os.path.join(seq_directory, f"{seq_name}_metrics_table.csv")
                if os.path.exists(csv_file):
                    try:
                        import pandas as pd
                        existing_df = pd.read_csv(csv_file)
                        for _, row in existing_df.iterrows():
                            tracker_combo = (row['Tracker'], row['Combo'])
                            existing_combinations.append(tracker_combo)
                    except Exception as e:
                        missing_components.append(f"CSV file corrupted: {e}")
                
                # Identify missing tracker/combination pairs
                missing_tracking = []
                for tracker_name in trackers:
                    for combo in combinations:
                        if (tracker_name, combo) not in existing_combinations:
                            missing_tracking.append((tracker_name, combo))
                
                # Check video files if save_videos is enabled
                missing_videos = []
                if save_videos_flag:
                    for tracker_name in trackers:
                        for combo in combinations:
                            video_path = os.path.join(seq_directory, f"{tracker_name.lower()}_{combo}.avi")
                            if not os.path.exists(video_path):
                                missing_videos.append(f"{tracker_name.lower()}_{combo}.avi")
                
                # Check plot files
                missing_plots = []
                required_plots = [
                    f"{seq_name}_metrics_bar.png",
                    f"{seq_name}_eao_trends.png", 
                    f"{seq_name}_precision_vs_robustness.png",
                    f"{seq_name}_metrics_table.md"
                ]
                
                for plot_file in required_plots:
                    plot_path = os.path.join(seq_directory, plot_file)
                    if not os.path.exists(plot_path):
                        missing_plots.append(plot_file)
                
                return {
                    'missing_tracking': missing_tracking,
                    'missing_videos': missing_videos,
                    'missing_plots': missing_plots,
                    'existing_combinations': existing_combinations,
                    'is_complete': len(missing_tracking) == 0 and len(missing_videos) == 0 and len(missing_plots) == 0
                }
            
            # Analyze completion status
            completion_info = analyze_sequence_completion(seq, seq_dir, list(trackers_dict.keys()), combinations, save_videos)
            
            if completion_info['is_complete']:
                print(f"Skipping sequence {seq}: already completed (CSV + videos + plots)")
                
                # Load existing results from CSV for final consolidation
                try:
                    import pandas as pd
                    csv_file = os.path.join(seq_dir, f"{seq}_metrics_table.csv")
                    existing_df = pd.read_csv(csv_file)
                    
                    # Reconstruct results dictionary from CSV
                    results[seq] = {}
                    for _, row in existing_df.iterrows():
                        tracker_name = row['Tracker']
                        combo = row['Combo']
                        
                        if tracker_name not in results[seq]:
                            results[seq][tracker_name] = {}
                        
                        results[seq][tracker_name][combo] = {
                            'EAO': row['EAO'],
                            'Robustness': row['Robustness'],
                            'Precision': row['Precision'],
                            'TrackingTime': row['TrackingTime'],
                            'FPS': row['FPS'],
                            'NumFrames': row['NumFrames'],
                            'NumFailures': row['NumFailures']
                        }
                    
                    print(f"Loaded existing results for {seq}: {len(existing_df)} combinations")
                    
                except Exception as e:
                    print(f"Warning: Could not load existing results for {seq}: {e}")
                
                # Skip this sequence but update progress bar
                pbar.update(len(trackers_dict) * len(combinations))
                continue
            else:
                # Selective processing - only process missing components
                missing_tracking = completion_info['missing_tracking']
                missing_plots = completion_info['missing_plots']
                missing_videos = completion_info['missing_videos']
                
                print(f"Partial processing for {seq}:")
                if missing_tracking:
                    print(f"  Missing tracking: {len(missing_tracking)} combinations")
                if missing_videos:
                    print(f"  Missing videos: {len(missing_videos)} files")
                if missing_plots:
                    print(f"  Missing plots: {missing_plots}")
                
                # Load existing results first
                try:
                    if completion_info['existing_combinations']:
                        import pandas as pd
                        csv_file = os.path.join(seq_dir, f"{seq}_metrics_table.csv")
                        existing_df = pd.read_csv(csv_file)
                        
                        # Initialize results[seq] and load existing data
                        results[seq] = {}
                        for _, row in existing_df.iterrows():
                            tracker_name = row['Tracker']
                            combo = row['Combo']
                            
                            if tracker_name not in results[seq]:
                                results[seq][tracker_name] = {}
                            
                            results[seq][tracker_name][combo] = {
                                'EAO': row['EAO'],
                                'Robustness': row['Robustness'],
                                'Precision': row['Precision'],
                                'TrackingTime': row['TrackingTime'],
                                'FPS': row['FPS'],
                                'NumFrames': row['NumFrames'],
                                'NumFailures': row['NumFailures']
                            }
                        print(f"  Loaded existing results: {len(existing_df)} combinations")
                except Exception as e:
                    print(f"  Warning: Could not load existing results: {e}")
                    results[seq] = {}
            
            # Ensure results[seq] exists
            if seq not in results:
                results[seq] = {}
                
            gt_bboxes = seq_data['groundtruth_bboxes']
            initial_bbox = seq_data['initial_bbox']
            
            for tracker_name, tracker_create in trackers_dict.items():
                if tracker_name not in results[seq]:
                    results[seq][tracker_name] = {}
                
                for combo in combinations:
                    # Skip if this combination already exists (selective processing)
                    if tracker_name in results[seq] and combo in results[seq][tracker_name]:
                        print(f"  Skipping {tracker_name}/{combo}: already exists")
                        pbar.update(1)
                        continue
                    
                    print(f"  Processing {tracker_name}/{combo}: missing result")
                    
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
            
            # Save results immediately after completing this sequence
            if seq in results and results[seq]:
                sequence_results = {seq: results[seq]}
                save_consolidated_results(sequence_results, results_dir, sequences=[seq])
                print(f"Saved results for sequence: {seq}")
                
                # Generate analysis plots for this sequence (always regenerate if missing)
                try:
                    if 'completion_info' in locals() and completion_info['missing_plots']:
                        print(f"Regenerating missing plots for sequence: {seq}")
                        for missing_plot in completion_info['missing_plots']:
                            print(f"  Regenerating: {missing_plot}")
                    else:
                        print(f"Generating analysis plots for sequence: {seq}")
                    
                    plot_metrics(results_dir, seq)
                    plot_eao_trends(results_dir, seq)
                    plot_precision_vs_robustness(results_dir, seq)
                    generate_metrics_table(results_dir, seq)
                    print(f"Analysis plots completed for sequence: {seq}")
                except Exception as e:
                    print(f"Warning: Failed to generate plots for {seq}: {e}")
            elif seq in results:
                # Even if no new tracking was done, regenerate missing plots
                try:
                    if 'completion_info' in locals() and completion_info['missing_plots']:
                        print(f"Regenerating missing plots for completed sequence: {seq}")
                        for missing_plot in completion_info['missing_plots']:
                            print(f"  Regenerating: {missing_plot}")
                        
                        plot_metrics(results_dir, seq)
                        plot_eao_trends(results_dir, seq)
                        plot_precision_vs_robustness(results_dir, seq)
                        generate_metrics_table(results_dir, seq)
                        print(f"Plot regeneration completed for sequence: {seq}")
                except Exception as e:
                    print(f"Warning: Failed to regenerate plots for {seq}: {e}")
    
    # Final consolidated save (in case any sequences were processed)
    if results:
        print("Creating final consolidated results...")
        save_consolidated_results(results, results_dir, sequences=sequence_list)
        print("Creating consolidated CSV for all sequences...")
        save_all_sequences_consolidated_csv(results, results_dir, sequences=sequence_list)
    
    # Clean up visualization
    if visualize:
        cv2.destroyAllWindows()
    
    # Print total evaluation time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"Total evaluation time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    
    return results