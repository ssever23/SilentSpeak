import asyncio, edge_tts


async def tts_to_file(text: str, voice: str, out_wav: str, rate="+0%", pitch="+0Hz"):
    
    if voice == 'en-US-FableTurboMultilingualNeural':
        pitch = "+25Hz"
    elif voice == 'en-US-AvaMultilingualNeural':
        pitch = '-15Hz'
        rate = '-5%'
    elif voice == 'en-US-RogerNeural':
        pitch = '+10Hz'
    elif voice == 'en-US-AvaNeural':
        pitch = '+13Hz'
    
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch)
    print(voice)
    with open(out_wav, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])


def speak_text(text, voice, out_wav):
    async def runner():
        await tts_to_file(text, voice, out_wav)
        return out_wav

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(runner())
    else:
        return asyncio.create_task(runner())
