# Set environment variables for optimal performance
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import numpy as np
import json, tempfile, subprocess
from tqdm import tqdm

# Custom function to check for occurence of lip movements
from pipeline.pre_check import has_lip_movement

# OpenCV for video processing
import cv2
cv2.setNumThreads(1)

# Mediapipe for facial landmarks
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

TARGET_FPS = 25           # target frames per second
ROI_SIZE = 88             # size of the square region of interest (mouth)
GRAYSCALE = True          # convert frames to grayscale
PADDING_SCALE = 1.6       # how much context around mouth (1.4–2.0 reasonable)
SMOOTH_WIN = 5            # moving average window (frames)
DETECT_MAX_SIDE = 720     # downscale for landmarking to save RAM/CPU

OUTPUT_DIR = Path("data/preprocessed_files/video_output")
if not os.path.exists(OUTPUT_DIR):
    print(f"Creating output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
else:
    print(f"Output directory already exists")

# Official mouth landmark indices (outer + inner lips)
LIPS_IDX = sorted(set([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
    14,  87, 178,  88, 95,  185,  40,  39,  37,  0,   267, 269, 270, 409,
    415, 310, 311, 312, 13,  82,  81,  42,  183, 78,  191, 80,  81,  82, 13
]))


def standardize_video(input_path, fps=TARGET_FPS):
    """Write a temp mp4 with fixed fps/pix_fmt. Avoid capturing huge stderr buffers."""
    tmp_out = os.path.join(tempfile.gettempdir(), "vsr_tmp_standardized.mp4")
    if os.path.exists(tmp_out): os.remove(tmp_out)
    cmd = [
        "ffmpeg","-hide_banner","-loglevel","error",
        "-y","-i",input_path,"-r",str(fps),"-an","-pix_fmt","yuv420p",tmp_out
    ]
    subprocess.check_call(cmd)
    return tmp_out


def clamp_box(x1,y1,x2,y2,W,H):
    return max(0,x1), max(0,y1), min(W,x2), min(H,y2)


def moving_average_np(arr, win=5):
    if win <= 1: return arr
    pad = win//2
    padded = np.pad(arr, ((pad,pad),(0,0)), mode='edge')
    csum = np.cumsum(padded, axis=0)
    sm = (csum[win:] - csum[:-win]) / float(win)
    # center-align by padding back to original length
    if len(sm) < len(arr):
        front = (len(arr)-len(sm))//2
        back  = len(arr)-len(sm)-front
        sm = np.pad(sm, ((front,back),(0,0)), mode='edge')
    return sm


# Safe video writer with multiple codec support
def safe_video_writer(path, fps, size, is_color):
    fourccs = [
        cv2.VideoWriter_fourcc(*"mp4v"),
        cv2.VideoWriter_fourcc(*"avc1"),
        cv2.VideoWriter_fourcc(*"XVID"),  # .avi fallback
    ]
    tried = []
    for fourcc in fourccs:
        writer = cv2.VideoWriter(path, fourcc, fps, size, isColor=is_color)
        if writer.isOpened():
            return writer, path
        tried.append(fourcc)
    # fallback to .avi if target was mp4
    if path.endswith(".mp4"):
        return safe_video_writer(path[:-4] + ".avi", fps, size, is_color)
    return None, None


# Preprocess video into frames
def preprocess_video(
    input_video_path,
    roi_size=ROI_SIZE,
    target_fps=TARGET_FPS,
    grayscale=GRAYSCALE,
    padding_scale=PADDING_SCALE,
    smooth_win=SMOOTH_WIN,
    detect_max_side=DETECT_MAX_SIDE,
    out_dir=OUTPUT_DIR,
    save_preview=True):
    
    if has_lip_movement(input_video_path):
        print("Sufficient lip movement detected.")
        pass
    else:
        return "No lip movement detected in video!"
        
    # 1) Standardize container/FPS
    std_path = standardize_video(input_video_path, fps=target_fps)

    # 2) Probe video
    cap = cv2.VideoCapture(std_path)
    assert cap.isOpened(), f"Could not open: {std_path}"
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # 3) First pass: run landmarks streaming; store *only* boxes
    lips_boxes = [None] * T
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as fm:
        cap = cv2.VideoCapture(std_path)
        idx = 0
        for _ in tqdm(range(T), desc="Pass 1/2: landmarks (streaming)"):
            ok, frame = cap.read()
            if not ok: break

            # Optional downscale for detection speed/memory
            h0, w0 = frame.shape[:2]
            scale = 1.0
            if max(h0, w0) > detect_max_side:
                scale = detect_max_side / float(max(h0, w0))
                frame_small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                frame_small = frame

            rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            try:
                res = fm.process(rgb)
            except Exception:
                res = None

            if res and res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                # map back to original coords if scaled
                pts = np.array([[lm[i].x * frame_small.shape[1] / scale,
                                 lm[i].y * frame_small.shape[0] / scale] for i in LIPS_IDX],
                               dtype=np.float32)
                x_min, y_min = pts[:,0].min(), pts[:,1].min()
                x_max, y_max = pts[:,0].max(), pts[:,1].max()
                cx, cy = (x_min+x_max)/2, (y_min+y_max)/2
                size = max(x_max-x_min, y_max-y_min) * padding_scale
                x1, y1 = int(cx - size/2), int(cy - size/2)
                x2, y2 = int(cx + size/2), int(cy + size/2)
                x1, y1, x2, y2 = clamp_box(x1,y1,x2,y2,W,H)
                lips_boxes[idx] = (x1,y1,x2,y2)
            # else None stays
            idx += 1
        cap.release()

    # 4) Fill gaps; fallback if no detections at all; smooth
    # forward/backward fill
    last = None
    for i in range(T):
        if lips_boxes[i] is None and last is not None:
            lips_boxes[i] = last
        elif lips_boxes[i] is not None:
            last = lips_boxes[i]
    last = None
    for i in range(T-1, -1, -1):
        if lips_boxes[i] is None and last is not None:
            lips_boxes[i] = last
        elif lips_boxes[i] is not None:
            last = lips_boxes[i]

    if all(b is None for b in lips_boxes):
        side = min(W, H)//3
        cx, cy = W//2, H//2
        lips_boxes = [(cx-side//2, cy-side//2, cx+side//2, cy+side//2)] * T

    boxes_np = np.array(lips_boxes, dtype=np.float32)  # (T,4)
    boxes_np = moving_average_np(boxes_np, win=smooth_win).astype(np.int32)

    # 5) Second pass: crop & write directly to disk (memmap + preview)
    stem = os.path.splitext(os.path.basename(input_video_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    out_npy = os.path.join(out_dir, f"{stem}_frames.npy")
    out_mmap  = os.path.join(out_dir, f"{stem}_frames.mmap")   # temp/raw streaming buffer
    out_meta = os.path.join(out_dir, f"{stem}_meta.json")
    out_preview = os.path.join(out_dir, f"{stem}_preview.mp4") if save_preview else None

    # create memmap: float32 [0,1], shape (T,H,W) or (T,H,W,3) if not grayscale
    if grayscale:
        mmap_shape = (T, roi_size, roi_size)
    else:
        mmap_shape = (T, roi_size, roi_size, 3)
    frames_mm = np.memmap(out_mmap, dtype=np.float32, mode='w+', shape=mmap_shape)

    # preview writer (streaming)
    writer = None
    if save_preview:
        writer, out_preview = safe_video_writer(out_preview, target_fps, (roi_size, roi_size), is_color=not grayscale)
        assert writer is not None, "Could not open preview writer"

    cap = cv2.VideoCapture(std_path)
    for i in tqdm(range(T), desc="Pass 2/2: crop+save (streaming)"):
        ok, frame = cap.read()
        if not ok:
            # Some files misreport T; stop early and use the frames we've written so far
            T = i
            break

        x1,y1,x2,y2 = boxes_np[i].tolist()
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((roi_size, roi_size, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (roi_size, roi_size), interpolation=cv2.INTER_AREA)

        if grayscale:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)  # (H,W)
            frames_mm[i, :, :] = (crop.astype(np.float32) / 255.0)
            if writer is not None:
                writer.write(crop)  # single channel OK with isColor=False
        else:
            frames_mm[i, :, :, :] = (crop.astype(np.float32) / 255.0)
            if writer is not None:
                writer.write(crop)

    cap.release()
    frames_mm.flush()
    del frames_mm
    if writer is not None:
        writer.release()

    # ---- finalize: convert raw .mmap -> real .npy with header ----
    # ACTUAL number of frames T (may be < reported due to read failure)
    final_shape = (int(T), roi_size, roi_size) if grayscale else (int(T), roi_size, roi_size, 3)

    # Reopen the raw buffer with the correct shape and save a proper .npy
    mm = np.memmap(out_mmap, dtype=np.float32, mode="r", shape=final_shape)
    
    np.save(out_npy, np.asarray(mm))   # creates header + data
    del mm

    # optional: remove the raw streaming buffer
    try:
        os.remove(out_mmap)
    except FileNotFoundError:
        pass

    # ---- save metadata (now points to the REAL .npy file) ----
    meta = {
        "source": input_video_path,
        "standardized": std_path,
        "fps": float(fps),
        "target_fps": target_fps,
        "roi_size": roi_size,
        "grayscale": grayscale,
        "padding_scale": padding_scale,
        "smooth_win": smooth_win,
        "num_frames": int(T),
        "shape": list(final_shape),
        "boxes_first_last": ([boxes_np[0].tolist(), boxes_np[T-1].tolist()] if T > 0 else None),
        "data_file": os.path.basename(out_npy),
        "format": "npy",
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    return out_npy, out_meta, out_preview