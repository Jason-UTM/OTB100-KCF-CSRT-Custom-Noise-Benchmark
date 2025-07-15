from services.noise import purge_noise_folders
from services.config import load_config

if __name__ == "__main__":
    # Load configuration
    CONFIG_PATH = r".\config.json"
    config = load_config(CONFIG_PATH)

    # Purge noise folders if needed
    purge_noise_folders(config['DATASET_PATH'])