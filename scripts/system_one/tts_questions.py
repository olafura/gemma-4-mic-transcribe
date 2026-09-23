"""Speak the `question` of each System One item into the WAV its `audio` names.

    ~/.local/share/pipx/venvs/edge-tts/bin/python scripts/system_one/tts_questions.py \
        data/system-one/spoken12/items.jsonl

Needs edge-tts (the library; its CLI does not start under Python 3.14) and
ffmpeg. The WAVs are 16 kHz mono s16, what `mix gemma.system_one generate`
reads, and are written beside the items file. Each item is spoken in its
`lang` (default English), alternating between two speakers per language so the
set is not one voice. Items that name the same WAV share it: it is spoken once.
`*.wav` is gitignored: the spoken set is regenerated from the items, not stored.
"""

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import edge_tts

VOICES = {
    "en": ["en-US-GuyNeural", "en-GB-SoniaNeural"],
    "de": ["de-DE-ConradNeural", "de-DE-KatjaNeural"],
    "es": ["es-ES-AlvaroNeural", "es-MX-DaliaNeural"],
    "fr": ["fr-FR-HenriNeural", "fr-FR-DeniseNeural"],
    "is": ["is-IS-GunnarNeural", "is-IS-GudrunNeural"],
    "it": ["it-IT-DiegoNeural", "it-IT-ElsaNeural"],
    "ja": ["ja-JP-KeitaNeural", "ja-JP-NanamiNeural"],
    "ko": ["ko-KR-InJoonNeural", "ko-KR-SunHiNeural"],
    "nl": ["nl-NL-MaartenNeural", "nl-NL-ColetteNeural"],
    "pl": ["pl-PL-MarekNeural", "pl-PL-ZofiaNeural"],
    "pt": ["pt-BR-AntonioNeural", "pt-BR-FranciscaNeural"],
    "zh": ["zh-CN-YunxiNeural", "zh-CN-XiaoxiaoNeural"],
}


async def speak(text, voice, wav):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        mp3 = Path(tmp.name)
    try:
        await edge_tts.Communicate(text, voice).save(str(mp3))
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3),
             "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(wav)],
            check=True,
        )
    finally:
        mp3.unlink(missing_ok=True)


async def main(items_path):
    items_path = Path(items_path)
    with items_path.open() as f:
        items = [json.loads(line) for line in f if line.strip()]
    seen = set()
    per_lang = {}
    for item in items:
        wav = items_path.parent / item["audio"]
        if wav in seen:
            continue
        seen.add(wav)
        lang = item.get("lang") or "en"
        index = per_lang.get(lang, 0)
        per_lang[lang] = index + 1
        if wav.exists():
            continue
        voices = VOICES[lang]
        await speak(item["question"], voices[index % len(voices)], wav)
        print(f"{wav.name}: {item['question']}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
