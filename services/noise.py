import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from typing import Union, List, Optional
from services.data import load_otb100_data

def apply_gaussian_noise(image: np.ndarray, mean: float = 0.0, std: float = 25.0) -> np.ndarray:
    """
    Apply Gaussian noise to an image.
    """
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def apply_salt_pepper_noise(image: np.ndarray, salt_prob: float = 0.01, pepper_prob: float = 0.01) -> np.ndarray:
    """
    Apply salt and pepper noise to an image.
    """
    noisy_image = image.copy()
    total_pixels = image.size
    salt_pixels = int(total_pixels * salt_prob)
    pepper_pixels = int(total_pixels * pepper_prob)
    
    salt_coords = [np.random.randint(0, i, salt_pixels) for i in image.shape[:2]]
    noisy_image[salt_coords[0], salt_coords[1]] = 255
    
    pepper_coords = [np.random.randint(0, i, pepper_pixels) for i in image.shape[:2]]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    
    return noisy_image

def apply_occlusion(image: np.ndarray, occlusion_level: float) -> np.ndarray:
    """
    Apply random occlusion to an image.
    """
    h, w = image.shape[:2]
    occlusion_area = int(h * w * occlusion_level)
    x = np.random.randint(0, w)
    y = np.random.randint(0, h)
    x_end = min(x + int(np.sqrt(occlusion_area)), w)
    y_end = min(y + int(np.sqrt(occlusion_area)), h)
    occluded_image = image.copy()
    occluded_image[y:y_end, x:x_end] = 0
    return occluded_image

def apply_noise_and_occlusion_to_sequences(
    dataset_path: str,
    sequences: Union[str, List[str], None] = None,
    noise_types: Optional[List[str]] = None,
    occlusion_levels: Optional[List[float]] = None
) -> None:
    """
    Apply noise and occlusion to specified OTB-100 sequences and save results.
    
    Args:
        dataset_path: Path to OTB-100 dataset
        sequences: Sequence name(s) to process (None for all)
        noise_types: List of noise types to apply (e.g., ['gaussian', 'salt_pepper'])
        occlusion_levels: List of occlusion levels to apply (e.g., [0.2, 0.4, 0.6])
    """
    if noise_types is None:
        noise_types = ['gaussian', 'salt_pepper']
    if occlusion_levels is None:
        occlusion_levels = [0.2, 0.4, 0.6]
    
    # Validate inputs
    if not noise_types or not all(isinstance(nt, str) for nt in noise_types):
        raise ValueError("noise_types must be a non-empty list of strings")
    if not occlusion_levels or not all(isinstance(ol, (int, float)) and 0 <= ol <= 1 for ol in occlusion_levels):
        raise ValueError("occlusion_levels must be a non-empty list of floats between 0 and 1")
    
    # Load sequence data
    data = load_otb100_data(dataset_path, sequences=sequences)
    if not data:
        raise ValueError(f"No valid sequences found in {dataset_path}")
    
    # Process each sequence
    for seq, seq_data in data.items():
        src_dir = os.path.join(dataset_path, seq, 'img')
        if not os.path.exists(src_dir):
            print(f"Warning: Source directory {src_dir} does not exist")
            continue
        
        frames = sorted([
            os.path.join(src_dir, f)
            for f in os.listdir(src_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if not frames:
            print(f"Warning: No valid frames found in {src_dir}")
            continue
        
        # Initialize progress bar
        total_tasks = len(frames) * len(noise_types) * len(occlusion_levels)
        with tqdm(total=total_tasks, desc=f"Processing {seq}", unit="frame") as pbar:
            for noise_type in noise_types:
                for occlusion_level in occlusion_levels:
                    # Create output directory
                    combo = f"{noise_type}_{occlusion_level}"
                    dst_dir = os.path.join(dataset_path, seq, f"img_{combo}")
                    
                    # Check if folder exists and has correct number of frames
                    if os.path.exists(dst_dir):
                        existing_frames = [
                            f for f in os.listdir(dst_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                        ]
                        if len(existing_frames) == len(frames):
                            print(f"Done for folder: {dst_dir} (already has {len(frames)} frames)")
                            # Update progress bar for skipped frames
                            pbar.update(len(frames))
                            continue
                        else:
                            print(f"Incomplete folder detected: {dst_dir} (has {len(existing_frames)}, needs {len(frames)})")
                            # Delete all contents and recreate
                            shutil.rmtree(dst_dir)
                            os.makedirs(dst_dir, exist_ok=True)
                    else:
                        os.makedirs(dst_dir, exist_ok=True)
                    
                    # Process each frame
                    for frame_path in frames:
                        frame = cv2.imread(frame_path)
                        if frame is None:
                            print(f"Warning: Failed to load frame {frame_path}")
                            pbar.update(1)
                            continue
                        
                        # Apply noise
                        if noise_type == 'gaussian':
                            frame = apply_gaussian_noise(frame)
                        elif noise_type == 'salt_pepper':
                            frame = apply_salt_pepper_noise(frame)
                        else:
                            print(f"Warning: Unknown noise type {noise_type}")
                            pbar.update(1)
                            continue
                        
                        # Apply occlusion
                        frame = apply_occlusion(frame, occlusion_level)
                        
                        # Save frame
                        dst_path = os.path.join(dst_dir, os.path.basename(frame_path))
                        cv2.imwrite(dst_path, frame)
                        pbar.update(1)

def purge_noise_folders(dataset_path):
    """
    Delete all folders except 'img' in each OTB-100 sequence directory, targeting
    noise/occlusion folders (e.g., 'img_gaussian_0.2').
    
    Args:
        dataset_path (str): Path to OTB-100 dataset directory (e.g., 'path/to/OTB100').
    
    Returns:
        None: Deletes folders and prints status messages.
    """
    # Get list of sequence directories
    sequence_dirs = [d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not sequence_dirs:
        print("Warning: No sequence directories found in", dataset_path)
        return
    
    # Initialize progress bar
    with tqdm(total=len(sequence_dirs), desc="Purging sequences", unit="sequence") as pbar:
        for seq in sequence_dirs:
            seq_path = os.path.join(dataset_path, seq)
            
            # Get list of subdirectories in sequence
            subdirs = [d for d in os.listdir(seq_path) 
                       if os.path.isdir(os.path.join(seq_path, d))]
            
            for subdir in subdirs:
                if subdir != 'img':  # Preserve original 'img' folder
                    folder_path = os.path.join(seq_path, subdir)
                    try:
                        shutil.rmtree(folder_path)
                        print(f"Deleted folder: {folder_path}")
                    except Exception as e:
                        print(f"Warning: Failed to delete {folder_path}: {e}")
            
            pbar.update(1)