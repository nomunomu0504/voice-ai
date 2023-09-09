import uvicorn

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from espnet2.bin.tts_inference import Text2Speech
import torch
import soundfile as sf
import uuid
import os

router = APIRouter()

fs, lang = 44100, "Japanese"
text2speech = Text2Speech.from_pretrained(
    model_tag="kan-bayashi/tsukuyomi_full_band_vits_prosody",
    device="cuda",  # or "cuda"
    speed_control_alpha=1.0,
    noise_scale=0.333,
    noise_scale_dur=0.333,
)


def TTS_streamer(text: str):
    with torch.no_grad():
        wav = text2speech(text)["wav"]
        filename = str(uuid.uuid4())
        sf.write(f"{filename}.wav", wav.view(-1).cpu().numpy(), text2speech.fs)
        with open(f"{filename}.wav", mode="rb") as wav_file:
            yield from wav_file
    os.remove(f"{filename}.wav")


@router.get("/tts")
async def tts_streamer(text: str):
    return StreamingResponse(TTS_streamer(text), media_type="audio/wav")

if __name__ == "__main__":
    config = uvicorn.Config("main:app", host="0.0.0.0",
                            port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()
