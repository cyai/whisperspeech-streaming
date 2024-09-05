import logging
import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from whisperspeech_streaming_server.core.libs import StreamingPipeline
from whisperspeech_streaming_server.core.schemas import (
    SynthesisRequest,
    SynthesisResponse,
)
from whisperspeech_streaming_server.utils.utils import split_sentence

from WhisperSpeech.whisperspeech.pipeline import Pipeline

router = APIRouter()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


async def send_audio_stream(websocket: WebSocket, audio_stream):
    try:
        async for audio_chunk in audio_stream:
            audio_bytes = audio_chunk.cpu().numpy().tobytes()
            response = SynthesisResponse(audio_chunk=audio_bytes)
            await websocket.send_bytes(response.audio_chunk)
    except WebSocketDisconnect:
        pass


async def tensor_to_async_iterable(tensor, chunk_size=1024):
    num_chunks = tensor.size(1) // chunk_size
    for i in range(num_chunks):
        yield tensor[:, i * chunk_size : (i + 1) * chunk_size]
    if tensor.size(1) % chunk_size != 0:
        yield tensor[:, num_chunks * chunk_size :]


class WebSocketHandler:
    def __init__(self, pipeline):
        # self.model = tts_model
        self.pipeline = pipeline
        self.connections = set()
        self.speaker = "../resources/Abdulla.mp3"

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.add(websocket)

    async def disconnect(self, websocket: WebSocket):
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()
            self.connections.remove(websocket)

    async def handle_websocket(self, websocket: WebSocket):
        await self.connect(websocket)
        try:
            # source_se = "checkpoints/base_speakers/EN/en_default_se.pth"
            while True:
                data = await websocket.receive_text()
                request = SynthesisRequest.parse_raw(data)
                if request.text == "":
                    await self.disconnect(websocket)
                    return
                text = request.text
                speaker = request.speaker
                language = request.language
                speed = request.speed
                logger.info(
                    f"Received text: {text}, speaker: {speaker}, language: {language}, speed: {speed}"
                )

                for texts in split_sentence(text):
                    for audio_stream in self.pipeline.generate(
                        texts, speaker=self.speaker, cps=10
                    ):
                        # print("DEBUG: audio_stream", audio_stream)
                        async_audio_stream = tensor_to_async_iterable(audio_stream)
                        await send_audio_stream(websocket, async_audio_stream)

                # await send_audio_stream(websocket, audio_stream)
        except WebSocketDisconnect:
            await self.disconnect(websocket)
        except Exception as e:
            logger.error(f"Error during text synthesis: {e}")
            await self.disconnect(websocket)


model_ref = "collabora/whisperspeech:s2a-q4-base-en+pl.model"

pipe = StreamingPipeline(s2a_ref=model_ref)

handler = WebSocketHandler(pipe)


@router.websocket("/synthesize")
async def synthesize(websocket: WebSocket):
    await handler.handle_websocket(websocket)
