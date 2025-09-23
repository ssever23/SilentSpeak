def pick_voice(gender: str, age_years: int, locale='en-US'):
    """
    Returns a Microsoft Edge TTS voice name keyed by locale, age band, and gender, all in English.
    """
    
    if age_years is None or gender not in ('male','female'):
        # safe fallback
        return f"{locale}-AriaNeural"  # neutral/friendly female

    if age_years <= 10:
        return f"{locale}-AnaNeural" if gender=='female' else f"{locale}-FableTurboMultilingualNeural"
    elif age_years <= 18:
        return f"{locale}-AvaNeural" if gender=='female' else f"{locale}-RogerNeural"
    elif age_years <= 25:
        return f"{locale}-JennyMultilingualNeural" if gender=='female' else f"{locale}-BrianMultilingualNeural"
    elif age_years <= 40:
        return f"{locale}-AriaNeural" if gender=='female' else f"{locale}-GuyNeural"
    elif age_years <= 65:
        return f"{locale}-EmmaNeural" if gender=='female' else f"{locale}-ChristopherNeural"
    else: 
        return f"{locale}-AvaMultilingualNeural" if gender=='female' else f"{locale}-DavisNeural"