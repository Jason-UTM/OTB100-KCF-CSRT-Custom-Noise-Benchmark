import json

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
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {config_path}: {e}")