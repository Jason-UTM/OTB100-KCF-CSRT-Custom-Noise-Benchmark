import os
import pandas as pd
from typing import Dict, List, Union

def save_tracker_results(
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    results_dir: str,
    sequences: Union[str, List[str], None] = None
) -> None:
    """
    Create directories for results and save tracking metrics as CSV files.
    
    Args:
        results: Dictionary with tracking results:
                 {sequence_name: {tracker_name: {noise_occlusion_combo: metrics}}}
        results_dir: Base directory to store results (e.g., 'results')
        sequences: Sequence name(s) to process (None for all in results)
    
    Creates:
        - Directory: results_dir/sequence_name (e.g., results/Basketball)
        - CSV files: results_dir/sequence_name/tracker_noise_occlusion.csv
                    (e.g., results/Basketball/kcf_salt_pepper_0.6.csv)
                    with headers: EAO, Robustness, Precision, TrackingTime, FPS, NumFrames, NumFailures
    """
    # Normalize sequences input
    if sequences is None:
        sequence_list = list(results.keys())
    elif isinstance(sequences, str):
        sequence_list = [sequences]
    else:
        sequence_list = sequences

    # Create base results directory
    os.makedirs(results_dir, exist_ok=True)

    # Define metrics headers
    metrics_headers = [
        'EAO', 'Robustness', 'Precision', 
        'TrackingTime', 'FPS', 'NumFrames', 'NumFailures'
    ]

    # Process each sequence
    for seq in sequence_list:
        if seq not in results:
            print(f"Warning: Sequence {seq} not found in results")
            continue

        # Create sequence directory
        seq_dir = os.path.join(results_dir, seq)
        os.makedirs(seq_dir, exist_ok=True)

        # Save results for each tracker and combination
        for tracker_name, tracker_results in results[seq].items():
            for combo, metrics in tracker_results.items():
                # Create CSV filename
                csv_filename = f"{tracker_name.lower()}_{combo}.csv"
                csv_path = os.path.join(seq_dir, csv_filename)

                # Prepare data as a single-row DataFrame
                data = {key: [metrics[key]] for key in metrics_headers}
                df = pd.DataFrame(data)

                # Save to CSV (overwrite if exists)
                if os.path.exists(csv_path):
                    print(f"Overwriting existing file: {csv_path}")
                else:
                    print(f"Creating new file: {csv_path}")
                df.to_csv(csv_path, index=False)

def save_consolidated_results(
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    results_dir: str,
    sequences: Union[str, List[str], None] = None
) -> None:
    """
    Create directories for results and save tracking metrics as consolidated CSV files.
    Each sequence gets one CSV file with all tracker and combination results.
    
    Args:
        results: Dictionary with tracking results:
                 {sequence_name: {tracker_name: {noise_occlusion_combo: metrics}}}
        results_dir: Base directory to store results (e.g., 'results')
        sequences: Sequence name(s) to process (None for all in results)
    
    Creates:
        - Directory: results_dir/sequence_name (e.g., results/Basketball)
        - CSV file: results_dir/sequence_name/sequence_name_metrics_table.csv
                   (e.g., results/Basketball/Basketball_metrics_table.csv)
                   with headers: Tracker,Combo,EAO,Robustness,Precision,TrackingTime,FPS,NumFrames,NumFailures
    """
    # Normalize sequences input
    if sequences is None:
        sequence_list = list(results.keys())
    elif isinstance(sequences, str):
        sequence_list = [sequences]
    else:
        sequence_list = sequences

    # Create base results directory
    os.makedirs(results_dir, exist_ok=True)

    # Define metrics headers
    headers = ['Tracker', 'Combo', 'EAO', 'Robustness', 'Precision', 
               'TrackingTime', 'FPS', 'NumFrames', 'NumFailures']

    # Process each sequence
    for seq in sequence_list:
        if seq not in results:
            print(f"Warning: Sequence {seq} not found in results")
            continue

        # Create sequence directory
        seq_dir = os.path.join(results_dir, seq)
        os.makedirs(seq_dir, exist_ok=True)

        # Prepare consolidated data
        rows = []
        for tracker_name, tracker_results in results[seq].items():
            for combo, metrics in tracker_results.items():
                row = [
                    tracker_name,
                    combo,
                    metrics['EAO'],
                    metrics['Robustness'],
                    metrics['Precision'],
                    metrics['TrackingTime'],
                    metrics['FPS'],
                    metrics['NumFrames'],
                    metrics['NumFailures']
                ]
                rows.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(rows, columns=headers)
        csv_filename = f"{seq}_metrics_table.csv"
        csv_path = os.path.join(seq_dir, csv_filename)

        # Save to CSV (overwrite if exists)
        if os.path.exists(csv_path):
            print(f"Overwriting existing consolidated file: {csv_path}")
        else:
            print(f"Creating new consolidated file: {csv_path}")
        df.to_csv(csv_path, index=False)