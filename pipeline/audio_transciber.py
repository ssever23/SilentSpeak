from faster_whisper import WhisperModel
from pipeline.pre_check import has_audio, detect_speech


def transcribe_speech_from_audio(video_path: str):
    
    if has_audio(video_path):
        pass
    else:
        return "Video file has no audio!"
    
    speech_check = detect_speech(video_path)
    if speech_check["has_speech"]:
        pass
    else:
        return "Video file doesn't contain speech!"
    
    model_size = "small"
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #compute_type = "float16" if device == "cuda" else "int8"
    compute_type = "int8"
    
    model = WhisperModel(model_size, device="cpu", compute_type=compute_type)

    # Transcribe
    segments, info = model.transcribe(
        video_path,
        beam_size=5,
        vad_filter=True,  # improve punctuation/wording on noisy audio
        vad_parameters=dict(min_silence_duration_ms=500),
        language=None,            # None = auto-detect
        condition_on_previous_text=True,  # better coherence
    )

    print(f"Detected language: {info.language} (prob={info.language_probability:.2f})")

    
    # Collect plain text and also save an SRT with timestamps
    all_text = []
    srt_lines = []
    for i, seg in enumerate(segments, start=1):
        all_text.append(seg.text)
        start = seg.start
        end = seg.end
        # SRT time format
        def t(s):
            h = int(s//3600); m = int((s%3600)//60); ss = s%60
            return f"{h:02}:{m:02}:{int(ss):02},{int((ss-int(ss))*1000):03}"
        srt_lines += [str(i), f"{t(start)} --> {t(end)}", seg.text.strip(), ""]

    
    # Write outputs
    all_text = "".join(all_text)
    
    return all_text