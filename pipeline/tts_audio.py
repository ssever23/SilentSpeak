import asyncio

import edge_tts
from edge_tts.exceptions import NoAudioReceived

async def tts_to_file(text: str, voice: str, out_wav: str, rate="+0%", pitch="+0Hz"):
    """Convert text to speech and save as WAV file."""
    
    if voice == "en-US-RogerNeural":
        pitch = "+10Hz"
    elif voice == "en-US-AvaNeural":
        pitch = "+13Hz"
    
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch)
    print(voice)

    audio_chunks = 0

    try:
        with open(out_wav, "wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks += 1
                    f.write(chunk["data"])
    except NoAudioReceived:
        preview = text[:120].replace("\n", " ")
        print(
            "TTS request returned no audio. "
            f"voice={voice}, rate={rate}, pitch={pitch}, text_len={len(text)}, "
            f"audio_chunks={audio_chunks}, text_preview={preview!r}"
        )
        raise

    if audio_chunks == 0:
        preview = text[:120].replace("\n", " ")
        print(
            "TTS stream completed without audio chunks. "
            f"voice={voice}, rate={rate}, pitch={pitch}, text_len={len(text)}, "
            f"text_preview={preview!r}"
        )

def speak_text(text, voice, out_wav):
    """Loads tts_to_file in an asyncio loop."""
    
    async def runner():
        await tts_to_file(text, voice, out_wav)
        return out_wav

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(runner())
    else:
        return asyncio.create_task(runner())
