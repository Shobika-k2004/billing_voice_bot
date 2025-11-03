# app.py
import os
import uuid
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Configure the Gemini API key
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    # Handle the error appropriately, maybe exit or log
    
# --- Temporary File Storage ---
# Create a temporary directory to store audio files before uploading
TEMP_DIR = "temp_audio"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)
    
    
# --- Frontend routes ---
@app.route('/', methods=['GET'])
def serve_index():
    return send_from_directory('.', 'index1.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

@app.route('/style.css', methods=['GET'])
def serve_css():
    return send_from_directory('.', 'style.css')

@app.route('/script.js', methods=['GET'])
def serve_js():
    return send_from_directory('.', 'script.js')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Receives audio data, sends it to Gemini, and returns the transcription."""
    print("\n--- New Transcription Request Received ---")
    data = request.get_json()
    if not data or 'audio' not in data:
        print("Error: No audio data in request.")
        return jsonify({'success': False, 'error': 'No audio data provided'}), 400

    audio_base64 = data['audio']
    language = data.get('languageCode', 'en-US')
    print(f"Received language code from frontend: {language}")

    # 1. Decode base64
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as e:
        print(f"Error decoding base64 audio: {e}")
        return jsonify({'success': False, 'error': f'Failed to process audio file: {e}'}), 500

    try:
        # 2. Ensure API key is configured
        if not os.getenv("GEMINI_API_KEY"):
            print("Error: GEMINI_API_KEY environment variable is not set.")
            return jsonify({'success': False, 'error': 'GEMINI_API_KEY is not set'}), 500

        # 3. Prepare audio data for the model
        audio_part = {"mime_type": "audio/webm", "data": audio_bytes}

        # 4. Prompt the Gemini model with the improved prompt
        model = genai.GenerativeModel(model_name='models/gemini-1.5-flash-latest')

        prompt = (
            "You are an expert ASR and translator. "
            f"The likely source language is {language}. "
            "Task: 1) Transcribe the speech. 2) Translate to fluent American English. "
            "Return ONLY the final American English text, well-punctuated and natural."
        )
        
        print("Sending request to Gemini API...")
        response = model.generate_content([prompt, audio_part])
        
        # Using resolve() is a good practice for some complex responses
        try:
            response.resolve()
        except Exception as e:
            print(f"Note: response.resolve() failed, but this might be okay. Error: {e}")
            pass
            
        transcription = (getattr(response, 'text', '') or '').strip()
        
        print(f"Received transcription from Gemini: '{transcription}'")

        if not transcription:
            print("Warning: Empty transcription received from model.")
            # Check for safety ratings or other reasons for a blocked response
            print(f"Full Gemini Response Parts: {response.parts}")
            print(f"Prompt Feedback: {response.prompt_feedback}")
            return jsonify({'success': False, 'error': 'Empty transcription received. The audio might be silent or the content was blocked.'}), 502

        return jsonify({'success': True, 'transcription': transcription})

    except Exception as e:
        print(f"CRITICAL Error during transcription with Gemini: {e}")
        return jsonify({'success': False, 'error': f'Gemini API error: {str(e)}'}), 500

    # The finally block for cleanup is no longer needed since we aren't saving a file

if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=port)