import base64
from concurrent.futures import ThreadPoolExecutor
import datetime
import time

import requests
import uvicorn
import streamlit as st

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from espnet2.bin.tts_inference import Text2Speech
import torch
import soundfile as sf
import uuid
import os
import re

# 音声合成の最小文字数。小さすぎると安定しない場合があります。
SPLIT_THRESHOLD = 4
# 息継ぎの秒数（s）
TIME_BUFFER = 0.1
# 待機中の実行スパン（s）
SLEEP_ITER = 0.2
TTS_ENDPOINT = 'https://nomunomu0504.ngrok.dev/tts'

app = FastAPI()

fs, lang = 44100, "Japanese"
text2speech = Text2Speech.from_pretrained(
    model_tag="kan-bayashi/tsukuyomi_full_band_vits_prosody",
    device="cpu",  # or "cuda"
    speed_control_alpha=1.0,
    noise_scale=0.333,
    noise_scale_dur=0.333,
)


def split_text(text: str):
    text_list = re.split('[\n、。]+', text)
    text_list_ = []
    for text in text_list:
        if text == '':
            continue
        if len(text) < SPLIT_THRESHOLD:
            try:
                text_list_[-1] = text_list_[-1] + '。' + text
            except IndexError:
                text_list_.append(text)
        else:
            text_list_.append(text)
    if len(text_list[0]) < SPLIT_THRESHOLD and len(text_list_) > 1:
        text_list_[1] = text_list[0] + '。' + text_list_[1]
        text_list_ = text_list_[1:]
    return text_list_


def TTS_streamer(text: str):
    print(f'TTS_streamer - 実行開始：{text}')
    with torch.no_grad():
        wav = text2speech(text)["wav"]
        filename = str(uuid.uuid4())
        sf.write(f"{filename}.wav", wav.view(-1).cpu().numpy(), text2speech.fs)
        with open(f"{filename}.wav", mode="rb") as wav_file:
            yield from wav_file
    os.remove(f"{filename}.wav")


def sound_player(response_content: str):
    # 参考：https://qiita.com/kunishou/items/a0a1a26449293634b7a0
    audio_placeholder = st.empty()
    audio_str = "data:audio/ogg;base64,%s" % (
        base64.b64encode(response_content).decode())
    audio_html = """
                    <audio autoplay=True>
                    <source src="%s" type="audio/ogg" autoplay=True>
                    Your browser does not support the audio element.
                    </audio>
                """ % audio_str

    audio_placeholder.empty()
    time.sleep(0.5)
    audio_placeholder.markdown(audio_html, unsafe_allow_html=True)


def get_tts_sound(text: str, url=TTS_ENDPOINT):
    print(f'get_tts_sound - 実行開始：text: {text}, url: {url}')
    params = {'text': text}
    response = requests.get(url, params=params)
    print(f'実行完了：{response}')
    return response.content, datetime.datetime.now()


@app.get("/tts")
async def tts_streamer(text: str):
    return StreamingResponse(TTS_streamer(text), media_type="audio/wav")


@app.get("/voice")
async def voice(text: str = None):
    if text is None:
        return {"error": "text is required"}

    executor = ThreadPoolExecutor(max_workers=2)
    futures = []
    split_response = split_text(text)

    for sq_text in split_response:
        future = executor.submit(get_tts_sound, sq_text)
        futures.append(future)

    block_time_list = [datetime.timedelta() for i in range(len(futures))]
    current_time = datetime.datetime.now()

    res_index = 0
    gap_time = datetime.timedelta()
    while res_index < len(futures):
        print(f'res_index: {res_index}')
        future = futures[res_index]
        print(f'future: {future}')
        if future.done():
            if res_index == 0:
                base_time = datetime.datetime.now()

            print(f'now: {datetime.datetime.now()}')
            print(f'base_time: {base_time}')
            print(f'block_time_list: {block_time_list}')
            if datetime.datetime.now() > base_time + block_time_list[res_index]:
                for i in range(len(block_time_list)):
                    if i > res_index:
                        # 音声長を計算。音声は32bitの16000Hz、base64エンコードの結果は1文字6bitの情報であるため、下記の計算で算出できます
                        block_time_list[i] += datetime.timedelta(seconds=(
                            len(future.result()[0])*6/32/16000)+gap_time.total_seconds()+TIME_BUFFER)
                print(f'実行完了：{split_response[res_index]}')
                print(
                    f'実行時間：{(future.result()[1] - current_time).total_seconds():.3f}s')
                print(f'音声の長さ：{len(future.result()[0])*6/32/16000:.3f}s')
                sound_player(future.result()[0])
                res_index += 1
                gap_time = datetime.timedelta()
        elif res_index != 0:
            gap_time += datetime.timedelta(seconds=SLEEP_ITER)
        time.sleep(SLEEP_ITER)
    executor.shutdown()

if __name__ == "__main__":
    config = uvicorn.Config(
        "main:app", host="0.0.0.0",
        port=8000, log_level="info"
    )
    server = uvicorn.Server(config)
    server.run()
