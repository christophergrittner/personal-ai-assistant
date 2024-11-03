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
import queue
import threading
import soundfile as sf

# Initialize logging and environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio settings
SAMPLE_RATE = 26000 # Original sample rate 24000
CHANNELS = 1
MAX_INT16 = 32768

# Add these to your existing audio settings
INPUT_SAMPLE_RATE = 26000  # Standard mic sample rate -- TODO: 44100
CHUNK_DURATION = 0.5  # Half second chunks
CHUNK_SAMPLES = int(INPUT_SAMPLE_RATE * CHUNK_DURATION)
SILENCE_THRESHOLD = 0.01

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

class AudioHandler:
    def __init__(self):
        self.recording = False
        self.audio_queue = queue.Queue()
        self.input_stream = None
        self.chunk_size = 32000
        self.gain = 5.0
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            logger.warning(f'Input status: {status}')
        if self.recording:
            # Debug logging
            audio_level = np.abs(indata).mean()
            logger.debug(f"Audio level: {audio_level}")
            
            # Store audio regardless of level (remove silence check for now)
            self.audio_queue.put(indata.copy())

    def start_recording(self):
        """Start recording from microphone"""
        try:
            self.recording = True
            self.audio_queue = queue.Queue()  # Clear the queue
            
            # List available devices
            devices = sd.query_devices()
            logger.info(f"Available audio devices: {devices}")
            
            # Select the desired input device by index or name
            device_index = 0  # Replace with the correct index for your microphone
            logger.info(f"Using input device index: {device_index}")
            
            self.input_stream = sd.InputStream(
                device=device_index,
                channels=1,
                samplerate=INPUT_SAMPLE_RATE,
                callback=self.audio_callback,
                blocksize=CHUNK_SAMPLES
            )
            self.input_stream.start()
            logger.info("Started recording...")
        except Exception as e:
            logger.error(f"Error starting recording: {e}")

    def stop_recording(self):
        """Stop recording and return the audio data"""
        try:
            self.recording = False
            if self.input_stream:
                self.input_stream.stop()
                self.input_stream.close()
            
            # Collect all audio data from queue
            audio_chunks = []
            logger.info(f"Queue size: {self.audio_queue.qsize()}")
            
            while not self.audio_queue.empty():
                audio_chunks.append(self.audio_queue.get())
            
            if not audio_chunks:
                logger.warning("No audio chunks collected")
                return None
                
            # Combine all chunks
            logger.info(f"Collected {len(audio_chunks)} audio chunks")
            audio_data = np.concatenate(audio_chunks)
            
            # Debug audio data
            logger.info(f"Audio data shape: {audio_data.shape}")
            logger.info(f"Audio data mean: {np.abs(audio_data).mean()}")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            logger.exception("Full stop error:")
            return None

    async def send_audio_chunks(self, websocket, audio_data):
        """Send audio data in smaller chunks"""
        print("Sending audio chunks...")
        print(audio_data.shape)
        print(audio_data.dtype)
        raise NotImplementedError("TODO:")

async def get_response(websocket):
    """Get and process the complete response from the assistant"""
    full_text = []
    
    try:
        logger.info("Waiting for response...")
        while True:
            response = await websocket.recv()
            response_data = json.loads(response)
            logger.info(f"Response content: {response_data}")
            response_type = response_data.get("type")
            logger.info(f"Received response type: {response_type}")
            
            # Log the entire response data for debugging
            logger.debug(f"Full response data: {response_data}")
            
            if response_type == "response.content_part.added":
                # Extract and append the text content
                content_part = response_data.get("part", {}).get("text", "")
                full_text.append(content_part)
                logger.info(f"Appending content part: {content_part}")
                print(content_part, end="", flush=True)
                
            elif response_type == "response.done":
                logger.info("Response completed.")
                print()  # New line after response
                break
                
            elif response_type == "error":
                error_msg = response_data.get("error", {}).get("message", "Unknown error")
                logger.error(f"API Error: {error_msg}")
                print(f"\nError: {error_msg}")
                break
                
    except Exception as e:
        logger.error(f"Error while getting response: {e}")
        logger.exception("Full traceback:")
        return "Sorry, there was an error getting the response."
        
    return "".join(full_text)

async def send_message(websocket, message, audio_data=None):
    """Send message and audio to websocket"""
    try:
        if audio_data:
            # Ensure audio data is base64 encoded
            print(f"Send message. Audio data shape: {audio_data.shape}")
            encoded_audio = base64.b64encode(audio_data).decode('utf-8')
            payload = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "audio": encoded_audio
                        }
                    ]
                }
            }
        else:
            # Handle text input
            payload = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": message
                        }
                    ]
                }
            }
        
        logger.info(f"Sending payload: {payload}")
        await websocket.send(json.dumps(payload))

        # Request response
        response_payload = {
            "type": "response.create"
        }
        await websocket.send(json.dumps(response_payload))
            
        logger.info("Message sent successfully")
    except Exception as e:
        logger.error(f"Error sending message: {e}")

def capture_audio_chunk():
    """Capture a chunk of audio data from the microphone."""
    try:
        # Record a chunk of audio
        audio_data = sd.rec(CHUNK_SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()  # Wait until the recording is finished
        return audio_data.flatten()  # Return as a 1D array
    except Exception as e:
        logger.error(f"Error capturing audio chunk: {e}")
        return np.array([])  # Return an empty array on error

def save_audio_as_wav(audio_data, sample_rate, filename="output.wav"):
    """Save the recorded audio data as a WAV file."""
    # Ensure audio_data is a 1D array
    if audio_data.ndim > 1:
        audio_data = audio_data.flatten()
    
    # Save the audio data as a WAV file
    sf.write(filename, audio_data, sample_rate)
    logger.info(f"Audio saved as {filename}")

async def main():
    uri = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "openai-beta": "realtime=v1"
    }
    
    audio_streamer = AudioStreamer()
    audio_handler = AudioHandler()
    
    try:
        async with websockets.connect(
            uri, 
            extra_headers=headers,
            ping_interval=20,
            ping_timeout=60
        ) as websocket:
            logger.info("Connected to WebSocket")
            print("Connected! Type your messages (or 'voice' to use voice input, 'exit' to quit)")
            
            while True:
                print("You: ", end="", flush=True)
                user_input = input()
                
                if user_input.lower() == 'exit':
                    break
                    
                if user_input.lower() == 'voice':
                    print("Recording... Press Enter to stop.")
                    audio_handler.start_recording()
                    input()  # Wait for Enter key
                    audio_data = capture_audio_data()  # Use the capture_audio_data function
                    audio_handler.stop_recording()
                    
                    # Save the audio data as a WAV for debugging
                    if audio_data is not None and audio_data.size > 0:
                        save_audio_as_wav(audio_data, INPUT_SAMPLE_RATE)
                        logger.info("Valid audio recorded, sending to API...")
                        await audio_handler.send_audio_chunks(websocket, audio_data)
                    else:
                        print("No audio recorded.")
                        continue
                else:
                    await send_message(websocket, user_input)
                
                print("Assistant: ", end="")
                response = await get_response(websocket)

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        logger.exception("Full traceback:")
    finally:
        audio_streamer.stop_stream()

if __name__ == "__main__":
    asyncio.run(main())