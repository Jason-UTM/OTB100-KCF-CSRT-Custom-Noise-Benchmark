import cv2

def create_tracker_by_name(name):
    if name == 'CSRT':
        return cv2.TrackerCSRT_create()
    elif name == 'KCF':
        return cv2.TrackerKCF_create()
    else:
        raise ValueError(f"Tracker '{name}' is not supported.")
        