import os

def get_otb100_sequences(dataset_path):
    """
    Get a list of OTB-100 dataset sequences.
    
    Args:
        dataset_path (str): Path to OTB-100 dataset directory (e.g., 'path/to/OTB100').
    
    Returns:
        list: List of sequence names (e.g., ['Basketball', 'Car1', 'Dog']).
    """
    sequences = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    return sequences

def load_otb100_data(dataset_path, sequences=None):
    """
    Load frame paths and ground-truth bounding boxes for OTB-100 dataset sequences.
    
    Args:
        dataset_path (str): Path to OTB-100 dataset directory (e.g., 'path/to/OTB100').
        sequences (str, list, or None): Sequence name (e.g., 'Basketball'), list of names (e.g., ['Car1', 'Dog']),
                                        or None for all sequences.
    
    Returns:
        dict: Dictionary with sequence names as keys and values as dictionaries containing:
              - 'frames': List of frame file paths (e.g., ['img/0001.jpg', ...]).
              - 'initial_bbox': Initial bounding box for first frame [x, y, width, height].
              - 'groundtruth_bboxes': List of bounding boxes for all frames.
    """
    # Initialize output dictionary
    data = {}
    
    # Normalize sequences input
    if isinstance(sequences, str):
        sequence_dirs = [sequences]  # Single sequence (e.g., 'Basketball')
    elif isinstance(sequences, list):
        sequence_dirs = sequences  # Multiple sequences
    else:
        # Load all sequences in dataset_path
        sequence_dirs = [d for d in os.listdir(dataset_path) 
                        if os.path.isdir(os.path.join(dataset_path, d))]
    
    # Process each sequence
    for seq in sequence_dirs:
        seq_path = os.path.join(dataset_path, seq)
        img_path = os.path.join(seq_path, 'img')
        
        # Check if sequence directory and img folder exist
        if not os.path.exists(img_path):
            print(f"Warning: Image folder not found for sequence {seq}")
            continue
        
        # Load frame paths (sorted to ensure correct order)
        frames = sorted([os.path.join(img_path, f) for f in os.listdir(img_path) 
                        if f.endswith(('.jpg', '.png'))])
        if not frames:
            print(f"Warning: No images found for sequence {seq}")
            continue
        
        # Load ground-truth bounding boxes
        gt_file = os.path.join(seq_path, 'groundtruth_rect.txt')
        if not os.path.exists(gt_file):
            print(f"Warning: Ground-truth file not found for sequence {seq}")
            continue
        
        bboxes = []
        with open(gt_file, 'r') as f:
            for line in f:
                # Handle comma-separated, space-separated, or tab-separated formats
                parts = line.strip().split(',') if ',' in line else line.strip().split()
                if len(parts) == 4:
                    bbox = [float(x) for x in parts]  # [x, y, width, height]
                    bboxes.append(bbox)
                else:
                    print(f"Warning: Invalid bounding box format in {gt_file}")
                    bboxes.append([0, 0, 0, 0])  # Placeholder for invalid lines
        
        # Ensure initial bounding box is valid
        initial_bbox = bboxes[0] if bboxes else [0, 0, 0, 0]
        
        # Store sequence data
        data[seq] = {
            'frames': frames,
            'initial_bbox': initial_bbox,
            'groundtruth_bboxes': bboxes
        }
    
    return data