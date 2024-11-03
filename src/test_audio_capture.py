import sounddevice as sd
import soundfile as sf
import numpy as np
import logging
import queue

# Initialize logging and audio settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 26000
CHANNELS = 1
CHUNK_SAMPLES = int(SAMPLE_RATE * 0.5)  # 0.5 seconds chunks

class AudioTester:
    def __init__(self):
        self.recording = False
        self.audio_queue = queue.Queue()
        self.input_stream = None

    def audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f'Input status: {status}')
        if self.recording:
            self.audio_queue.put(indata.copy())

    def start_recording(self):
        try:
            self.recording = True
            self.audio_queue = queue.Queue()
            
            # List available devices
            devices = sd.query_devices()
            logger.info("Available audio devices:")
            for i, device in enumerate(devices):
                logger.info(f"{i}: {device['name']} (in: {device['max_input_channels']}, out: {device['max_output_channels']})")
            
            # Select device with input capabilities
            device_index = None
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    device_index = i
                    break
            
            if device_index is None:
                raise ValueError("No input device found")
                
            logger.info(f"Using input device index: {device_index}")
            
            self.input_stream = sd.InputStream(
                device=device_index,
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                callback=self.audio_callback,
                blocksize=CHUNK_SAMPLES
            )
            self.input_stream.start()
            logger.info("Started recording...")
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            raise

    def stop_recording(self):
        try:
            self.recording = False
            if self.input_stream:
                self.input_stream.stop()
                self.input_stream.close()
            
            # Collect audio data from queue
            audio_chunks = []
            while not self.audio_queue.empty():
                audio_chunks.append(self.audio_queue.get())
            
            if not audio_chunks:
                logger.warning("No audio chunks collected")
                return None
                
            audio_data = np.concatenate(audio_chunks)
            logger.info(f"Audio data shape: {audio_data.shape}")
            return audio_data
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return None

def save_audio_file(audio_data, filename="test_recording.wav"):
    """Save the audio data to a WAV file"""
    try:
        sf.write(filename, audio_data, SAMPLE_RATE)
        logger.info(f"Audio saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving audio: {e}")

def main():
    tester = AudioTester()
    
    try:
        input("Press Enter to start recording...")
        tester.start_recording()
        input("Recording... Press Enter to stop.")

        audio_data = tester.stop_recording()
        save_audio_file(audio_data)
        print(f"Recording completed and saved!")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        if tester.input_stream:
            tester.input_stream.close()

if __name__ == "__main__":
    main() 