import os
import json
import websockets
import asyncio
from dotenv import load_dotenv
import logging

# Initialize environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def send_message(websocket, text):
    # Create the message event
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
    
    # Send the message and request response
    await websocket.send(json.dumps(message_event))
    await websocket.send(json.dumps({"type": "response.create"}))

async def main():
    uri = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "openai-beta": "realtime=v1"
    }
    
    try:
        async with websockets.connect(uri, extra_headers=headers) as websocket:
            logger.info("Connected to WebSocket")
            print("Connected! Type your messages (type 'exit' to quit)")
            
            while True:
                # Get user input
                user_input = input("You: ")
                if user_input.lower() == 'exit':
                    break
                
                # Send message
                await send_message(websocket, user_input)
                
                # Initialize response collection
                print("Assistant: ", end="")
                
                # Listen for responses
                while True:
                    response = await websocket.recv()
                    response_data = json.loads(response)
                    
                    # Print only the delta text without the "Assistant:" prefix
                    if response_data.get("type") == "response.audio_transcript.delta":
                        print(response_data.get("delta"), end="", flush=True)
                    elif response_data.get("type") == "response.done":
                        print("\n")  # Add newline after response is complete
                        break

    except Exception as e:
        logger.error(f"Error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 