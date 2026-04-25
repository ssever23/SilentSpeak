# SilentSpeak
Visual Speech Recognition (VSR) models are capable of detecting a large proportion of spoken words from a video of a person speaking. However, the resulting transcriptions are typically incoherent, lack punctuation and contain significant errors. To address this limitation, this project introduces a pipeline that integrates a Large Language Model (LLM) to refine the raw VSR output, leveraging the semantic and contextual understanding power of LLMs to reconstruct more accurate and meaningful text.

The full pipeline processes the lip movements of a single visible speaker in a video and generates natural-sounding audio output. It works in five main steps:

1. **Lip Movement Detection** – Checks whether the video contains sufficient visible lip movement.

2. **Visual Speech Recognition** – Uses a VSR model to predict text from the speaker's lip movements.

3. **LLM Refinement** – Cleans, punctuates, and semantically reconstructs the raw VSR output, improving clarity, coherence, and accuracy.

4. **Voice Synthesis** – Estimates the speaker's age and gender, selects a matching voice, and generates a natural-sounding audio output as a `.wav` file.

5. **Audio Comparison** – If the input video contains audio, transcribes it and compares it with the VSR-derived text.

The result is an end-to-end system that converts the lip movements from a video into clear, lifelike speech, while also supporting validation against original audio when available.

![](https://github.com/ssever23/SilentSpeak/blob/main/VSR%20pipeline.jpg)

## Example
The example below uses an [actor’s monologue](https://www.youtube.com/watch?v=YDhszbGqBmk). It compares the original audio transcript, the raw VSR output, and the LLM-refined reconstruction.

| Version | Source | Description |
| --- | --- | --- |
| Original audio transcript | Video audio | Reference transcript used for comparison |
| Raw VSR output | Lip movements only | Direct output from the visual speech recognition model |
| LLM-refined output | Raw VSR text only | Cleaned and reconstructed text generated without access to the video or audio |

<details>
<summary><strong>Original audio transcript</strong></summary>

You understand? Yeah, I mean, of course you'd say that. Such a typical thing for a therapist to say. You know what? Let's just cut the crap, alright? I'm not stupid. I know that you could care less about me or my problems. And all you really care about is getting that money at the end of this session. So you know what? Let's make a deal. You know, I'm fine with just sitting here an hour a week on my phone. You can do whatever it is that you do. Then my mom will be happy because I'm here and you'll be happy because you get your money. Now do we have a deal? Or would you rather just keep playing this fake sympathetic therapist that pretends to care about her clients and I'll just go along being the gullible patient who thinks that I finally have someone who wants to help me. Either way, you get your money and I understand that's all you really want. And scene.

</details>

<details>
<summary><strong>Raw VSR output</strong></summary>

you understand yeah i mean of course you can say that such a typical thing for the top so you say oh you know what let's just on the crap right i'm not stupid i know that you can hear less time to care less time about me who would be problems and all you really care about is that that money at that discussion so you know what you know what let's make a dealer i'm fine when she's sitting here and how are we on my phone you can do whatever it's like you do i think my mom will be happy because the mirror and and you'll be happy because it gets your money and i also have a deal or would you rather just complain this finger simulating therapist that pretends to care about her clients and all just going on me and the carbon patient who thinks that i finally have someone who wants to help me wants to help me in a way you get your money and you say that's all you really want really was i said

</details>

<details>
<summary><strong>LLM-refined output</strong></summary>

You understand? Yeah, I mean, of course you can say that. It's such a typical thing for a therapist to say. You know what? Let's just cut the crap. I'm not stupid. I know you couldn't care less about me or my problems. All you really care about is the money from this session. So, let's make a deal. I'll sit here on my phone, and you can do whatever it is you do. My mom will be happy because I'm here, and you'll be happy because you get your money. So, do we have a deal? Or would you rather we just continue this charade? You, playing the therapist who pretends to care about her clients, and me, the gullible patient who thinks I've finally found someone who wants to help me. Either way, you get your money. That's all you really want, isn't it?

</details>

### Result

The word error rate (WER) between the original speech transcript and the raw VSR output is **58.24%**.
After refinement with the LLM, that number decreases to **38.24%**.

Although the WER remains relatively high after refinement, it primarily reflects syntactic mismatches rather than differences in meaning. A semantic comparison reveals that while the raw VSR output diverges substantially from the original speech, the refined version preserves its core meaning and intent almost entirely. This suggests that LLM-based refinement can help turn noisy visual speech recognition output into coherent and meaningful text, even when the result is not an exact transcript.

## How to run

1. Create an `ext` named folder in the project root and clone the [AV-HuBERT repository](https://github.com/facebookresearch/av_hubert) into it:

```bash
git clone https://github.com/facebookresearch/av_hubert.git
```

2. Place an AV-HuBERT checkpoint in the `model/` directory. AV-HuBERT checkpoints can be downloaded from the official [AV-HuBERT checkpoint page](https://facebookresearch.github.io/av_hubert/). By default, the model path in main.py looks for:

```text
model/base_vox_433h.pt
```

3. Create a `.env` file in the project root and add your Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
```

The current implementation uses Gemini for LLM refinement. You can use another LLM provider, but you will likely need to update the API call in `pipeline/llm_refiner.py`. The refinement prompt itself can be kept and reused.

4. Run the pipeline from the project root:

```bash
python3 -m pipeline.main
```

5. When prompted, enter the path to the input video:

```text
Enter video path: data/input_video/example.mp4
```

The generated text output is saved to:

```text
data/output_files/vsr_text.txt
```

If you choose to generate speech audio, the `.wav` file is saved to:

```text
data/output_files/speech_audio.wav
```

## Limitations

SilentSpeak currently works best with videos that contain:

- one clearly visible speaker
- frontal or near-frontal face positioning
- sufficient lighting and mouth visibility
- minimal occlusion, motion blur, or profile view

The LLM refinement step improves readability and semantic coherence, but it may also infer words that were not present in the original speech.

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
