from services.benchmark import run_trackers_and_evaluate
from services.config import load_config

if __name__ == "__main__":
    # Load configuration
    CONFIG_PATH = r".\config.json"
    config = load_config(CONFIG_PATH)

    # Extract parameters
    DATASET_PATH = config['DATASET_PATH']
    RESULTS_DIR = config['RESULTS_DIR']
    SEQUENCES = config['SEQUENCES']
    TRACKERS = config['TRACKERS']
    VISUALIZE = config['VISUALIZE']
    SAVE_VIDEOS = config['SAVE_VIDEOS']
    FAILURE_THRESHOLD = config['FAILURE_THRESHOLD']
    FRAME_DELAY = config['FRAME_DELAY']

    # Run trackers and evaluate
    print("Running trackers and evaluating...")
    results = run_trackers_and_evaluate(
        dataset_path=DATASET_PATH,
        results_dir=RESULTS_DIR,
        sequences=SEQUENCES,
        trackers=TRACKERS,
        visualize=VISUALIZE,
        save_videos=SAVE_VIDEOS,
        failure_threshold=FAILURE_THRESHOLD,
        frame_delay=FRAME_DELAY
    )
    
    # Print results
    for seq in results:
        for tracker in results[seq]:
            for combo in results[seq][tracker]:
                print(f"{seq}/{tracker}/{combo}: {results[seq][tracker][combo]}")