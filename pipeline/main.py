import os
from pathlib import Path

from pipeline.preprocessing import preprocess_video
from pipeline.lip_reader import predict_speech
from pipeline.llm_refiner import vsr_text

from pipeline.gender_age_identifier import estimate_gender_and_age
from pipeline.voice_picker import pick_voice
from pipeline.tts_audio import speak_text

from pipeline.audio_transciption import transcribe_speech_from_audio


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "base_vox_433h.pt"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output_files"
TEXT_OUTPUT_PATH = OUTPUT_DIR / "vsr_text.txt"
AUDIO_OUTPUT_PATH = OUTPUT_DIR / "speech_audio.wav"


def main():
    video_path = input("Enter video path: ")
    
    # Preprocess video by extracting and storing lip movements
    preprocess_result = preprocess_video(video_path)
    if preprocess_result == 0:
        print("No sufficient lip movement detected!")
        return 

    npy_file, _, _ = preprocess_result

    # Generate speech text through lip movements
    lip_text = predict_speech(model_path=str(MODEL_PATH), npy_path=npy_file)

    # Refine lip reading text with LLM
    api_key = os.getenv("GEMINI_API_KEY")
    llm_model = "gemini-2.5-pro"
    speech = vsr_text(lip_text, api_key, llm_model)

    # Transcribed speech from video audio for comparison
    reference_text = transcribe_speech_from_audio(video_path)

    vsr_output = (
    f"Transcription from audio:\n\n{reference_text}\n\n"
    f"Text from lip movements:\n\n{lip_text}\n\n"
    f"Refined text by LLM:\n\n{speech}\n"
    )

    # Identify gender and age of person speaking
    _, age_median, gender = estimate_gender_and_age(video_path)

    ga = f"\nGender and age of person speaking: {gender}, {age_median}\n\n"

    out_txt = ga + vsr_output

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with TEXT_OUTPUT_PATH.open('w', encoding='utf-8') as f:
        f.write(out_txt)

    ask_speech_audio = input("Would you like a voiced audio file of the generated speech [yes|no]: ")

    if ask_speech_audio.lower() == 'yes':
        # Create speech audio
        # Pick voice based on gender and age
        voice = pick_voice(gender=gender, age_years=age_median)
        # Turn VSR text into spoken speech
        speech_audio = speak_text(text=speech, voice=voice, out_wav=str(AUDIO_OUTPUT_PATH))

        audio_path = f"Speech audio file saved under: {speech_audio}"

        return print(f"\nTexts saved under: {TEXT_OUTPUT_PATH}\n\n"
                    f"{audio_path}")
    else:
        return print(f"\nTexts saved under: {TEXT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
