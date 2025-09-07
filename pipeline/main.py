import os
from dotenv import load_dotenv
load_dotenv()

from pipeline.preprocessing import preprocess_video
from pipeline.lip_reader import predict_speech
from pipeline.llm_refiner import vsr_text

from pipeline.gender_age_identifier import estimate_gender_and_age
from pipeline.voice_picker import pick_voice
from pipeline.tts_audio import speak_text

from pipeline.audio_transciption import transcribe_speech_from_audio


def main():
    
    video_path = input("Enter video path: ")
    
    # Preprocess video by extracting and storing lip movements
    npy_file = preprocess_video(video_path)[0]
    
    # Identify speech by lip movements
    model_path = "/home/ssever/SilentSpeak/model/base_vox_433h.pt"
    lip_text = predict_speech(model_path=model_path, npy_path=npy_file)
    
    # Correct lip reading text with LLM
    api_key = os.getenv("GEMINI_API_KEY")
    llm_model = "gemini-2.5-pro"
    speech = vsr_text(lip_text, api_key, llm_model)
    
    # Transcribed speech from video audio for comparison
    reference_text = transcribe_speech_from_audio(video_path)
    
    # Output spoken speech
    # Identify gender and age of person speaking
    _, age_median, gender = estimate_gender_and_age(video_path)
    # Pich voice based on gender and age
    voice = pick_voice(gender=gender, age_years=age_median)
    # Turn VSR text into spoken speech
    out_wav = "/home/ssever/SilentSpeak/data/output_files/speech.wav"
    speak = speak_text(text=speech, voice=voice, out_wav=out_wav)
    
    ga = f"\nGender and age of person speaking: {gender}, {age_median}\n\n"
    
    vsr_output = (
    f"Transcription from audio:\n\n{reference_text}\n\n"
    f"Text from lip movements:\n\n{lip_text}\n\n"
    f"Cleaned text by LLM:\n\n{speech}\n"
    )
    
    out_txt = ga + vsr_output
    
    file_path = '/home/ssever/SilentSpeak/data/output_files/vsr_text.txt'
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(out_txt)
    
    audio_path = f"Speech audio file saved under: {speak}"
    
    return print(f"\nTexts saved under: {file_path}\n\n"
                 f"{audio_path}")


if __name__ == "__main__":
    main()