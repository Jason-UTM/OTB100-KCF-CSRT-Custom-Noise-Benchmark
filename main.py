from services.config import load_config
from services.noise import apply_noise_and_occlusion_to_sequences, purge_noise_folders
from services.benchmark import run_trackers_and_evaluate
from services.analysis import plot_metrics, plot_eao_trends, plot_precision_vs_robustness

if __name__ == "__main__":
    # Load configuration
    CONFIG_PATH = r".\config.json"
    config = load_config(CONFIG_PATH)

    # Extract parameters
    DATASET_PATH = config['DATASET_PATH']
    RESULTS_DIR = config['RESULTS_DIR']
    SEQUENCES = config['SEQUENCES']
    TRACKERS = config['TRACKERS']
    NOISE_TYPES = config['NOISE_TYPES']
    OCCLUSION_LEVELS = config['OCCLUSION_LEVELS']
    VISUALIZE = config['VISUALIZE']
    SAVE_VIDEOS = config['SAVE_VIDEOS']
    FAILURE_THRESHOLD = config['FAILURE_THRESHOLD']
    FRAME_DELAY = config['FRAME_DELAY']

    # Generate noisy frames
    print("Generating noisy frames...")
    apply_noise_and_occlusion_to_sequences(
        dataset_path=DATASET_PATH,
        sequences=SEQUENCES,
        noise_types=NOISE_TYPES,
        occlusion_levels=OCCLUSION_LEVELS
    )

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

    # Analyze results
    for sequence in SEQUENCES:
        print(f"Analyzing results for sequence: {sequence}")
        plot_metrics(RESULTS_DIR, sequence)
        plot_eao_trends(RESULTS_DIR, sequence)
        plot_precision_vs_robustness(RESULTS_DIR, sequence)
