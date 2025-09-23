# ====================================
# Section 1: Checking for lip movement
# ====================================

import cv2
import numpy as np
import mediapipe as mp

# Inner/outer lips landmark indices (MediaPipe FaceMesh, 468 points).
# Sources summarizing lip indices and FaceMesh basics: MediaPipe docs + community maps.
UPPER_LIPS = [13, 82, 81, 42, 183, 78]      # upper inner/near-inner band (includes mid 13)
LOWER_LIPS = [14, 87, 178, 88, 95]          # lower inner/near-inner band (includes mid 14)
MOUTH_CORNERS_INNER = (78, 308)             # inner corners (more stable for width than 61/291)


def _xy_from_landmarks(landmarks, w, h, idx):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def _mouth_width(landmarks, w, h):
    L = _xy_from_landmarks(landmarks, w, h, MOUTH_CORNERS_INNER[0])
    R = _xy_from_landmarks(landmarks, w, h, MOUTH_CORNERS_INNER[1])
    return float(np.linalg.norm(R - L) + 1e-6)


def _aperture_from_many_pairs(landmarks, w, h):
    """
    Robust mouth opening:
    - Builds sets of upper/lower lip points.
    - Pairs each upper point to the closest-by-x lower point.
    - Takes the median vertical gap and normalize by inner-corner width.
    """
    width = _mouth_width(landmarks, w, h)
    if width <= 1e-6:
        return 0.0

    upp = np.array([_xy_from_landmarks(landmarks, w, h, i) for i in UPPER_LIPS])
    low = np.array([_xy_from_landmarks(landmarks, w, h, i) for i in LOWER_LIPS])

    # Pair by nearest x (prevents relying on exact index correspondences)
    gaps = []
    for u in upp:
        j = np.argmin(np.abs(low[:,0] - u[0]))  # nearest x
        vgap = abs(low[j,1] - u[1])            # vertical distance
        gaps.append(vgap)
    if not gaps:
        return 0.0

    return float(np.median(gaps) / width)


def has_lip_movement(
    video_path: str,            
    sample_fps: float = 5.0,           # sample rate for the precheck
    min_face_fraction: float = 0.2,    # require faces in at least 20% of sampled frames
    min_modulation_std: float = 0.015, # require some variance over time
    adapt_k: float = 0.6,              # how far above baseline we call it "open"
    min_open_fraction: float = 0.20    # require at least 20% of face frames to be “open”  -> i.e. person is talking
):
    
    print("Checking that there is sufficient lip movement first!")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return dict(ok=False, reason=f"Could not open {video_path}")
    
    #video_duration = _get_duration(video_path)
    video_duration = int(45)

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames_to_scan = int(min(video_duration * native_fps, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1e9))
    step = max(int(round(native_fps / sample_fps)), 1)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,         # << better lip detail
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    sampled, face_frames = 0, 0
    apertures = []
    frame_idx = 0
    while frame_idx < total_frames_to_scan:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        sampled += 1

        if res.multi_face_landmarks:
            face_frames += 1
            lms = res.multi_face_landmarks[0].landmark
            apertures.append(_aperture_from_many_pairs(lms, w, h))

        frame_idx += step

    cap.release()
    try:
        face_mesh.close()
    except Exception:
        pass

    if sampled == 0:
        return dict(ok=False, reason="No frames sampled")

    face_fraction = face_frames / sampled
    if face_fraction < min_face_fraction:
        return dict(
            ok=False,
            reason=f"Face too infrequent: {face_fraction:.2f} < {min_face_fraction}",
            face_fraction=face_fraction
        )

    if len(apertures) < 5:
        return dict(ok=False, reason="Too few lip samples", face_fraction=face_fraction)

    a = np.array(apertures, dtype=np.float32)
    # Smooth slightly to suppress per-frame jitter
    if len(a) >= 5:
        a = np.convolve(a, np.ones(5)/5.0, mode="same")

    # Adaptive threshold: baseline = low quantile; high = 90th quantile
    base = float(np.quantile(a, 0.20))   # ~closed mouth level
    hi   = float(np.quantile(a, 0.90))   # very open
    thr  = base + adapt_k * max(hi - base, 1e-6)

    open_ratio = float(np.mean(a > thr))
    a_std = float(a.std())

    talking_like = (a_std >= min_modulation_std) and (open_ratio >= min_open_fraction)
    return dict(
        ok=talking_like,
        reason=("Face present and speech-like mouth motion" if talking_like
                else f"Not speech-like: open_ratio={open_ratio:.2f} (min {min_open_fraction}), std={a_std:.3f} (min {min_modulation_std})"),
        face_fraction=face_fraction,
        open_ratio=open_ratio,
        aperture_std=a_std,
        thr=thr,
        stats=dict(min=float(a.min()), med=float(np.median(a)), p90=hi, base=base)
    )


# ========================================
# Section 2: Checking for audio and speech
# ========================================

import subprocess
import json
import webrtcvad


def has_audio(video_path: str) -> bool:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "json", video_path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    info = json.loads(result.stdout)
    return "streams" in info and len(info["streams"]) > 0


def _pcm_stream(input_media: str, sr: int = 16000, bandpass: bool = True):
    """
    Yields raw PCM16 mono audio bytes from input_media via ffmpeg.
    Optional band-pass ~100-3800 Hz applied before piping to VAD.
    """
    af = []
    if bandpass:
        # Tight speech band to cut rumble/aircon & bright SFX
        af.append("highpass=f=100")
        af.append("lowpass=f=3800")
    af_str = ",".join(af) if af else "anull"

    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", input_media, "-vn", "-ac", "1", "-ar", str(sr),
        "-af", af_str,
        "-f", "s16le", "pipe:1"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            yield chunk
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()
        

def detect_speech(
    path: str,
    aggressiveness: int = 3,      # most strict
    frame_ms: int = 10,           # 10 ms frames reduce false positives
    min_speech_ms: int = 1200,    # require sustained speech
    min_speech_ratio: float = 0.05,
    min_consec_frames: int = 12,  # at least ~120 ms continuous speech
    sample_rate: int = 16000
):
    """
    Returns a dict with has_speech + stats, using stricter rules to avoid SFX.
    """
    if frame_ms not in (10, 20, 30):
        raise ValueError("frame_ms must be 10, 20, or 30.")

    vad = webrtcvad.Vad(aggressiveness)

    bytes_per_sample = 2
    frame_bytes = int(sample_rate * (frame_ms / 1000.0)) * bytes_per_sample

    total_frames = 0
    speech_frames = 0
    consec = 0
    consec_hits = 0  # number of times it hits >= min_consec_frames

    buffer = bytearray()
    for chunk in _pcm_stream(path, sr=sample_rate, bandpass=True):
        buffer.extend(chunk)
        # exact framing
        while len(buffer) >= frame_bytes:
            frame = bytes(buffer[:frame_bytes])
            del buffer[:frame_bytes]

            total_frames += 1
            is_sp = vad.is_speech(frame, sample_rate)
            if is_sp:
                speech_frames += 1
                consec += 1
                if consec == min_consec_frames:
                    consec_hits += 1  # count a sustained run
            else:
                consec = 0  # reset streak on non-speech

    speech_ratio = (speech_frames / total_frames) if total_frames else 0.0
    min_speech_frames = int(min_speech_ms / frame_ms)

    # Final decision must pass ALL gates
    has_speech = (
        speech_frames >= min_speech_frames and
        speech_ratio >= min_speech_ratio and
        consec_hits >= 2   # saw at least two ~70ms sustained runs
    )

    return {
        "has_speech": has_speech,
        "speech_frames": speech_frames,
        "total_frames": total_frames,
        "speech_ratio": speech_ratio,
        "consecutive_runs": consec_hits
    }