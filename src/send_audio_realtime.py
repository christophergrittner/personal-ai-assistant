import asyncio
import websockets
import numpy as np
import base64
import json
import os
from test_audio_capture import AudioTester, SAMPLE_RATE, save_audio_file

def float_to_pcm16(float32_array):
    """Convert Float32Array to PCM16 ArrayBuffer (matches floatTo16BitPCM in JS)"""
    float32_array = np.clip(float32_array, -1, 1)
    pcm16_data = (float32_array * 32767).astype(np.int16)
    return pcm16_data.tobytes()

def base64_encode_audio(float32_array):
    """Convert Float32Array to base64-encoded PCM16 data"""
    pcm16_buffer = float_to_pcm16(float32_array)
    chunk_size = 0x8000  # 32KB chunks, matching JS implementation
    base64_chunks = []
    
    for i in range(0, len(pcm16_buffer), chunk_size):
        chunk = pcm16_buffer[i:i + chunk_size]
        base64_chunks.append(base64.b64encode(chunk).decode())
    
    return ''.join(base64_chunks)

async def send_audio_to_realtime():
    uri = "wss://api.openai.com/v1/realtime"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "openai-beta": "realtime=v1"
    }
    
    async with websockets.connect(uri, extra_headers=headers) as websocket:
        tester = AudioTester()
        
        try:
            input("Press Enter to start recording...")
            tester.start_recording()
            input("Recording... Press Enter to stop.")
            
            audio_data = tester.stop_recording()
            save_audio_file(audio_data)
            
            if audio_data is not None and audio_data.size > 0:
                # Add debug logging before sending
                print(f"Audio data shape: {audio_data.shape}")
                print(f"Sample rate: {SAMPLE_RATE}")
                print(f"Audio duration: {len(audio_data) / SAMPLE_RATE:.2f} seconds")
                
                # Convert audio data to float32 (-1 to 1 range)
                float32_data = audio_data.astype(np.float32) / 32767.0
                base64_audio = base64_encode_audio(float32_data)
                
                # Create event matching JS structure exactly
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
                
                # Send event and response.create
                await websocket.send(json.dumps(event))
                await websocket.send(json.dumps({"type": "response.create"}))
                print("Audio data sent successfully")
            else:
                print("No audio data captured")
            
        except Exception as e:
            print(f"Error: {e}")
        finally:
            if tester.input_stream:
                tester.input_stream.close()

if __name__ == "__main__":
    asyncio.run(send_audio_to_realtime())