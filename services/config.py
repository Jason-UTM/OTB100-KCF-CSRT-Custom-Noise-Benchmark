import json
import os

required_keys = [
    'DATASET_PATH', 
    'RESULTS_DIR', 
    'SEQUENCES', 
    'TRACKERS',
    'NOISE_TYPES', 
    'OCCLUSION_LEVELS', 
    'VISUALIZE',
    'SAVE_VIDEOS', 
    'FAILURE_THRESHOLD', 
    'FRAME_DELAY'
]

def load_config(config_path: str) -> dict:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the JSON configuration file.
    
    Returns:
        Dictionary containing configuration parameters.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        # Validate required keys
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
        # Handle empty sequences - process all available sequences
        sequences = config['SEQUENCES']
        if not sequences or sequences == [""] or (isinstance(sequences, list) and all(not seq.strip() for seq in sequences if isinstance(seq, str))):
            dataset_path = config['DATASET_PATH']
            if os.path.exists(dataset_path):
                all_sequences = [
                    d for d in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, d))
                ]
                # Filter out sequences without ground-truth files
                valid_sequences = []
                excluded_sequences = []
                for seq in all_sequences:
                    gt_path = os.path.join(dataset_path, seq, 'groundtruth_rect.txt')
                    if os.path.exists(gt_path):
                        valid_sequences.append(seq)
                    else:
                        excluded_sequences.append(seq)
                
                config['SEQUENCES'] = valid_sequences
                print(f"Empty sequences detected. Processing all available sequences with ground-truth: {valid_sequences}")
                if excluded_sequences:
                    print(f"Excluded sequences without ground-truth files: {excluded_sequences}")
            else:
                print(f"Warning: Dataset path {dataset_path} not found. Using empty sequences list.")
                config['SEQUENCES'] = []
        else:
            # Validate manually specified sequences have ground-truth files
            dataset_path = config['DATASET_PATH']
            if os.path.exists(dataset_path):
                valid_sequences = []
                excluded_sequences = []
                for seq in sequences:
                    if seq.strip():  # Skip empty strings
                        gt_path = os.path.join(dataset_path, seq, 'groundtruth_rect.txt')
                        if os.path.exists(gt_path):
                            valid_sequences.append(seq)
                        else:
                            excluded_sequences.append(seq)
                
                config['SEQUENCES'] = valid_sequences
                if excluded_sequences:
                    print(f"Warning: Excluded manually specified sequences without ground-truth files: {excluded_sequences}")
                    print(f"Processing only valid sequences: {valid_sequences}")
        
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {config_path}: {e}")