import os
import json
import websockets
import asyncio
import logging
from dotenv import load_dotenv
from test_audio_capture import AudioTester, save_audio_file
import numpy as np
import sounddevice as sd
import base64

# Initialize logging and environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio settings for playback (as per API documentation)
SAMPLE_RATE = 24000  # API uses 24kHz for output

async def get_response(websocket):
    """Get and process the complete response from the assistant"""
    full_text = []
    
    try:
        logger.info("Waiting for response...")
        while True:
            response = await websocket.recv()
            response_data = json.loads(response)
            response_type = response_data.get("type")
            
            if response_type == "error":
                error_msg = response_data.get("error", {}).get("message", "Unknown error")
                logger.error(f"API Error: {error_msg}")
                print(f"\nError: {error_msg}")
                break
                
            elif response_type == "response.text.delta":
                text_delta = response_data.get("delta", "")
                full_text.append(text_delta)
                print(text_delta, end="", flush=True)
                
            elif response_type == "response.audio.delta":
                # Decode and play audio chunk
                audio_data = base64.b64decode(response_data.get("delta", ""))
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Play audio
                sd.play(audio_array, SAMPLE_RATE)
                # Wait for just enough time to prevent overlap
                duration = len(audio_array) / SAMPLE_RATE
                await asyncio.sleep(duration * 0.6)  # Reduced to 10% of chunk duration
                
            elif response_type == "response.done":
                print()  # New line after response
                logger.info("Response completed")
                # Minimal wait for final audio
                await asyncio.sleep(0.3)
                break

            elif response_type != "response.audio_transcript.delta":
                logger.info(f"{response_type}")
                logger.info(response)
                print("________________________\n")
                
    except Exception as e:
        logger.error(f"Error while getting response: {e}")
        return "Sorry, there was an error getting the response."
        
    return "".join(full_text)

async def update_session(websocket):
    update_message = {
        "type": "session.update",
        "session": {
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "voice": "sage",
            "modalities": ["text", "audio"],
            "instructions": "Please be concise and direct in your responses",
        }
    }
    await websocket.send(json.dumps(update_message))
    response = await websocket.recv()
    print(f"Received response: {response}")

def base64_encode_audio(float32_array):
    """Convert Float32Array to base64-encoded PCM16 data"""
    pcm16_buffer = float_to_pcm16(float32_array)
    chunk_size = 0x8000  # 32KB chunks, matching JS implementation
    base64_chunks = []
    
    for i in range(0, len(pcm16_buffer), chunk_size):
        chunk = pcm16_buffer[i:i + chunk_size]
        base64_chunks.append(base64.b64encode(chunk).decode())
    
    return ''.join(base64_chunks)

def float_to_pcm16(float32_array):
    """Convert Float32Array to PCM16 ArrayBuffer (matches floatTo16BitPCM in JS)"""
    float32_array = np.clip(float32_array, -1, 1)
    pcm16_data = (float32_array * 32767).astype(np.int16)
    return pcm16_data.tobytes()

async def main():
    uri = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "openai-beta": "realtime=v1"
    }
    
    async with websockets.connect(uri, extra_headers=headers) as websocket:
        await update_session(websocket)
        tester = AudioTester()
        
        try:
            while True:
                input("Press Enter to start recording...")
                tester.start_recording()
                input("Recording... Press Enter to stop.")
                
                audio_data = tester.stop_recording()
                if audio_data is not None and audio_data.size > 0:
                    # Save audio file for reference
                    save_audio_file(audio_data)
                    
                    # Convert and send audio data
                    # Convert float32 to int16 (PCM16) format
                    pcm16_data = (audio_data * 32767).astype(np.int16)
                    base64_audio = base64_encode_audio(pcm16_data)

                    # Create and send audio event
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
                    logger.info(f"Audio submission response: {audio_response}")
                    
                    # Request a response with both text and audio
                    response_event = {
                        "type": "response.create",
                        "response": {
                            "modalities": ["text", "audio"],
                            "instructions": "Please assist the user."
                        }
                    }
                    await websocket.send(json.dumps(response_event))
                    
                    # Get and display response
                    print("\nAssistant: ", end="")
                    response = await get_response(websocket)
                    print(response)
                else:
                    print("No audio data captured.")
                    
        except Exception as e:
            logger.error(f"Error occurred: {e}")
        finally:
            if tester.input_stream:
                tester.input_stream.close()

if __name__ == "__main__":
    asyncio.run(main())