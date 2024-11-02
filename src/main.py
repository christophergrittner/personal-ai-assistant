import os
import json
import websockets
import asyncio
from dotenv import load_dotenv
import logging
import sounddevice as sd
import numpy as np
import base64
from collections import deque
import time

# Initialize logging and environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio settings
SAMPLE_RATE = 26000 # Original sample rate 24000
CHANNELS = 1
MAX_INT16 = 32768

class AudioStreamer:
    def __init__(self):
        self.queue = deque()
        self.stream = None
        self.is_playing = False
        
    def callback(self, outdata, frames, time, status):
        """Callback for the sounddevice stream"""
        if status:
            logger.warning(f'Audio status: {status}')
        
        if len(self.queue) == 0:
            outdata.fill(0)
            if self.is_playing:
                self.is_playing = False
            return
            
        # Get next chunk of audio
        data = self.queue.popleft()
        if len(data) < len(outdata):
            outdata[:len(data)] = data.reshape(-1, 1)
            outdata[len(data):] = 0
        else:
            outdata[:] = data[:len(outdata)].reshape(-1, 1)
            # Put remaining data back in queue
            if len(data) > len(outdata):
                self.queue.appendleft(data[len(outdata):])
        
        self.is_playing = True
    
    def start_stream(self):
        """Start the audio stream"""
        try:
            if self.stream is None or not self.stream.active:
                self.stream = sd.OutputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    callback=self.callback,
                    blocksize=1024,  # Smaller blocksize for more responsive playback
                    latency='low'    # Request low latency
                )
                self.stream.start()
                self.is_playing = True
        except Exception as e:
            logger.error(f"Error starting stream: {e}")
    
    def stop_stream(self):
        """Stop the audio stream"""
        try:
            # Wait for queue to empty
            while self.is_playing and len(self.queue) > 0:
                time.sleep(0.1)
            
            if self.stream is not None and self.stream.active:
                self.stream.stop()
                self.stream.close()
            self.stream = None
            self.is_playing = False
        except Exception as e:
            logger.error(f"Error stopping stream: {e}")
            self.stream = None
    
    def add_audio(self, audio_data: str):
        """Add new audio data to the queue"""
        try:
            decoded = base64.b64decode(audio_data)
            audio_array = np.frombuffer(decoded, dtype=np.int16)
            normalized = audio_array.astype(np.float32) / MAX_INT16
            self.queue.append(normalized)
            
            # Start stream if it's not already running
            if self.stream is None or not self.stream.active:
                self.start_stream()
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")

async def get_response(websocket, audio_streamer):
    """Get and process the complete response from the assistant"""
    full_text = []
    
    try:
        while True:
            response = await websocket.recv()
            response_data = json.loads(response)
            
            if response_data.get("type") == "response.audio_transcript.delta":
                # Handle text
                delta = response_data.get("delta", "")
                full_text.append(delta)
                print(delta, end="", flush=True)
                
            elif response_data.get("type") == "response.audio.delta":
                # Handle audio in real-time
                audio_data = response_data.get("delta", "")
                audio_streamer.add_audio(audio_data)
                
            elif response_data.get("type") == "response.done":
                print()  # New line after response
                # Wait a bit longer for audio to finish
                await asyncio.sleep(1.0)
                audio_streamer.stop_stream()
                break
                
    except Exception as e:
        logger.error(f"Error while getting response: {e}")
        return "Sorry, there was an error getting the response."
        
    return "".join(full_text)

async def send_message(websocket, text):
    """Send a message to the assistant"""
    message_event = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": text
                }
            ]
        }
    }
    
    await websocket.send(json.dumps(message_event))
    await websocket.send(json.dumps({"type": "response.create"}))

async def main():
    uri = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "openai-beta": "realtime=v1"
    }
    
    audio_streamer = AudioStreamer()
    
    try:
        async with websockets.connect(uri, extra_headers=headers) as websocket:
            logger.info("Connected to WebSocket")
            print("Connected! Type your messages (type 'exit' to quit)")
            
            while True:
                user_input = input("You: ")
                if user_input.lower() == 'exit':
                    break
                
                await send_message(websocket, user_input)
                print("Assistant: ", end="")
                response = await get_response(websocket, audio_streamer)

    except Exception as e:
        logger.error(f"Error occurred: {e}")
    finally:
        audio_streamer.stop_stream()

if __name__ == "__main__":
    asyncio.run(main())