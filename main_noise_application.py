from services.config import load_config
from services.noise import apply_noise_and_occlusion_to_sequences

if __name__ == "__main__":
    # Load configuration
    CONFIG_PATH = r".\config.json"
    config = load_config(CONFIG_PATH)

    # Extract parameters
    DATASET_PATH = config['DATASET_PATH']
    SEQUENCES = config['SEQUENCES']
    NOISE_TYPES = config['NOISE_TYPES']
    OCCLUSION_LEVELS = config['OCCLUSION_LEVELS']

    # Generate noisy frames
    print("Generating noisy frames...")
    apply_noise_and_occlusion_to_sequences(
        dataset_path=DATASET_PATH,
        sequences=SEQUENCES,
        noise_types=NOISE_TYPES,
        occlusion_levels=OCCLUSION_LEVELS
    )