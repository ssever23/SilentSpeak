# SilentSpeak
This project processes videos of a single person speaking and generates natural-sounding audio output. The pipeline works in four main steps:

1. **Speech Extraction** – Lip movements are analyzed to extract the spoken text.

2. **Text Refinement** – An LLM cleans and improves the extracted text for readability and accuracy.

3. **Voice Synthesis** – The speaker’s age and gender are estimated to select an appropriate voice, which is then used to generate the final audio file.

4. **Audio Comparison** – If the input video already contains speech audio, it is transcribed to compare with the extracted text.

The result is an end-to-end system that converts silent video into clear, lifelike speech, while also supporting validation against original audio when available. Before running the pipeline, the program first checks for the presence of lip movements to ensure processing is only triggered when someone is actually speaking.

![](https://github.com/ssever23/SilentSpeak/blob/main/VSR%20pipeline.png)

## AV-HuBERT

This repository incorporates [AV-HuBERT](https://github.com/facebookresearch/av_hubert) for lip reading. AV-HuBERT is a self-supervised representation learning framework for audio-visual speech. It achieves state-of-the-art results in lip reading, ASR and audio-visual speech recognition on the LRS3 audio-visual speech benchmark. AV-HuBERT is licensed under the AV-HuBERT license, Copyright (c) Meta Platforms, Inc. All Rights Reserved.

## License

AV-HuBERT LICENSE AGREEMENT

This License Agreement (as may be amended in accordance with this License
Agreement, “License”), between you (“Licensee” or “you”) and Meta Platforms,
Inc. (“Meta” or “we”) applies to your use of any computer program, algorithm,
source code, object code, or software that is made available by Meta under this
License (“Software”) and any specifications, manuals, documentation, and other
written information provided by Meta related to the Software (“Documentation”).

By using the Software, you agree to the terms of [this
License](https://github.com/ssever23/SilentSpeak?tab=License-1-ov-file). If
you do not agree to this License, then you do not have any rights to use the
Software or Documentation (collectively, the “Software Products”), and you must
immediately cease using the Software Products.