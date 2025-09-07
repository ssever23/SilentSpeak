import cv2, numpy as np
from insightface.app import FaceAnalysis


def estimate_gender_and_age(video_path, sample_stride=5):
    """
    Returns (age_mean, age_median, gender_mode, stats_dict) from visual frames.
    - sample_stride: analyze every Nth frame for speed
    - gender: 'male' or 'female' (string), based on per-frame argmax logits
    """
    app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'genderage'])  # good default bundle
    app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=-1 for CPU, 0 for GPU
    
    # Load video file
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open {video_path}"
    
    ages, genders = [], []
    frame_idx = 0

    # Read every N-th frame
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_stride != 0:
            frame_idx += 1
            continue
    
        # Detect the most central face.
        faces = app.get(frame)
        if faces:
            # get image center
            H, W = frame.shape[:2]
            cx, cy = W/2, H/2
            # sort faces by acending order from distance to center
            faces.sort(key=lambda f: ( (f.bbox[0]+f.bbox[2])/2 - cx )**2 + ( (f.bbox[1]+f.bbox[3])/2 - cy )**2)
            # pick face closest to the center
            f0 = faces[0]
            # determine age
            if f0.age is not None:
                ages.append(float(f0.age))
            # determine gender
            if f0.gender is not None:
                genders.append('male' if int(f0.gender) == 1 else 'female')

        frame_idx += 1
        
    cap.release()
    
    #if len(ages) < max(5, min_frames):
    #    return None, None, None, {"frames_analyzed": len(ages)}
    
    
    age_arr = np.array(ages, dtype=float)
    # robust smoothing: median is less sensitive; also clip to plausible range
    age_median = int(np.clip(np.median(age_arr), 5, 90)) if ages else None
    age_mean = int(np.clip(np.mean(age_arr), 5, 90)) if ages else None
    
    # pick gender that was detected most often
    gender = max(set(genders), key=genders.count) if genders else None

    return age_mean, age_median, gender
    
    