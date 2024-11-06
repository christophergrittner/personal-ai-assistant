import asyncio
import websockets
import json
import os
import sounddevice as sd
import numpy as np
import base64
from dotenv import load_dotenv
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
load_dotenv()

# Audio settings
SAMPLE_RATE = 16000  # Input sample rate
OUTPUT_RATE = 24000  # Output sample rate
audio_queue = queue.Queue()

def play_audio():
    """Play audio from the queue"""
    while True:
        try:
            audio_data = audio_queue.get()
            if audio_data is None:  # Sentinel value to stop the thread
                break
            # Convert base64 to numpy array and play
            audio_bytes = base64.b64decode(audio_data)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            sd.play(audio_np, samplerate=OUTPUT_RATE)
            sd.wait()  # Wait until audio is finished playing
        except Exception as e:
            print(f"Error playing audio: {e}")

async def main():
    # Start audio playback thread
    audio_thread = threading.Thread(target=play_audio)
    audio_thread.start()

    # Record audio
    duration = 5  # seconds
    print("Recording")
    audio_data = sd.rec(int(duration * SAMPLE_RATE), 
                       samplerate=SAMPLE_RATE, 
                       channels=1, 
                       dtype=np.float32)
    sd.wait()
    print("\nFinished recording")

    # Debug logging
    print(f"Audio shape: {audio_data.shape}")
    print(f"Audio dtype: {audio_data.dtype}")
    print(f"Audio min/max: {audio_data.min()}, {audio_data.max()}")

    # Convert float32 to int16 PCM
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # Convert to base64 string
    audio_bytes = audio_int16.tobytes()
    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

    uri = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "openai-beta": "realtime=v1"
    }

    async with websockets.connect(uri, extra_headers=headers) as websocket:
        # Update session
        await websocket.send(json.dumps({
            "type": "session.update",
            "session": {
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "sample_rate": SAMPLE_RATE,
                "voice": "alloy",
                "modalities": ["text", "audio"],
                "instructions": "Please be concise and direct in your responses",
            }
        }))
        response = await websocket.recv()
        print(f"Session update response: {response}")

        # Send audio
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "audio": base64_audio
                    }
                ]
            }
        }
        
        await websocket.send(json.dumps(event))
        audio_response = await websocket.recv()
        print(f"Audio submission response: {audio_response}")

        # Request response
        await websocket.send(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"]
            }
        }))

        # Get response
        while True:
            response = await websocket.recv()
            response_data = json.loads(response)
            
            # Handle audio deltas
            if response_data.get("type") == "response.audio.delta":
                audio_queue.put(response_data["delta"])
            
            # Print text transcript
            elif response_data.get("type") == "response.audio_transcript.delta":
                print(response_data["delta"], end="", flush=True)
            
            # Check if response is complete
            elif response_data.get("type") == "response.done":
                break

        # Signal audio thread to stop
        audio_queue.put(None)
        audio_thread.join()

if __name__ == "__main__":
    asyncio.run(main())

