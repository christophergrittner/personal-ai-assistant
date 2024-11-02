import os
import json
import websockets
import asyncio
from dotenv import load_dotenv

# Initialize environment variables from .env file
load_dotenv()

async def send_message():
    # Connect to OpenAI's WebSocket endpoint
    uri = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "openai-beta": "realtime=v1"
    }
    
    # Establish WebSocket connection with OpenAI
    async with websockets.connect(uri, extra_headers=headers) as websocket:
        # Prepare the message payload
        # This follows OpenAI's realtime API event format:
        # - type: specifies the event type
        # - item: contains message details including role and content
        message_event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Hello!"
                    }
                ]
            }
        }
        
        # Send the message to OpenAI
        await websocket.send(json.dumps(message_event))
        
        # Request OpenAI to generate a response
        await websocket.send(json.dumps({"type": "response.create"}))
        
        # Listen for the response
        while True:
            response = await websocket.recv()
            print(f"Received: {response}")

async def main():
    try:
        await send_message()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 