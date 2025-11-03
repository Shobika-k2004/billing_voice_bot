import sounddevice as sd
import numpy as np
import queue
import threading
import asyncio
import edge_tts
import os
import tempfile
import argparse
import groq
import wave
from playsound import playsound
import webrtcvad
import collections
import argostranslate.package
import argostranslate.translate
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Advanced Real-time Voice Assistant with Groq Whisper")
    parser.add_argument("--language", type=str, default=None, help="Language code for transcription (e.g., 'en', 'ta'). Leave blank for auto-detection.")
    parser.add_argument("--vad_aggressiveness", type=int, default=3, choices=[0, 1, 2, 3], help="VAD aggressiveness (0=least, 3=most aggressive)")
    parser.add_argument("--silence_duration_ms", type=int, default=1500, help="Milliseconds of silence to consider the end of a phrase")
    return parser.parse_args()

args = parse_args()

# ---------------------------
# --- DOMAIN RESTRICTION SETUP (KEYWORD-BASED) ---
# ---------------------------
BILLING_KEYWORDS = {"billing","revenue","amount","pat count","billtype","regnumber","ipno","patientname",
"dept","admitdoctor","ledger","groupname","headername","orderingdoctor","orderingdept","servicename",
"time","billno","servqty","taxamount","concamt","city","cityid","district",
"state","serviceid","iheader_id","igroup_id","iadmit_doc_id","ord_doc_id","ideptid","ord_dept_id",
"patid","opid","ipid","pricelistid","mrd","userid","pat_type_id","patient type","rate","corperate type",
"corp_type_id","bedno","ward","dynamic amount range","bedtype","financialyear","qtr","quarter","avgrpb",
"discount","docconcamt","net","bed category","standard financial year","date level label","weekday","dr","rev","city names","last year","last month"}

# ---------------------------
# 2. Initialize Models & Tools
# ---------------------------
print("Initializing Groq client...")
client = groq.Groq(api_key="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")
print("Groq client ready.")

# Initialize Voice Activity Detection (VAD)
vad = webrtcvad.Vad(args.vad_aggressiveness)

# Translation model setup
print("Checking/downloading translation models...")
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()

def get_or_install_translator(from_code, to_code="en"):
    try:
        translator = argostranslate.translate.get_translation_from_codes(from_code, to_code)
        if translator:
            return translator
    except Exception:
        pass
    package_to_install = next(filter(lambda x: x.from_code == from_code and x.to_code == to_code, available_packages), None)
    if package_to_install:
        print(f"Downloading and installing translator: {from_code} -> {to_code}")
        argostranslate.package.install_from_path(package_to_install.download())
        return argostranslate.translate.get_translation_from_codes(from_code, to_code)
    else:
        return None

print("Pre-loading common translators...")
LANGS_TO_PRELOAD = ["hi", "ta", "fr", "es", "de"]
for lang in LANGS_TO_PRELOAD:
    get_or_install_translator(lang, "en")
print("Translators ready.")

# ---------------------------
# 3. Setup Audio Recording & VAD
# ---------------------------
q = queue.Queue()
samplerate = 16000
channels = 1
frame_duration_ms = 30
frame_size = int(samplerate * frame_duration_ms / 1000)

devices = sd.query_devices()
input_devices = [i for i, d in enumerate(devices) if d["max_input_channels"] > 0]
if not input_devices:
    raise RuntimeError("No input microphone found!")
mic_index = input_devices[0]
print(f"Using mic: {devices[mic_index]['name']} (index {mic_index})")

def audio_callback(indata, frames, time_, status):
    if status:
        print(status)
    q.put(bytes(indata))

# ---------------------------
# 4. Text-to-Speech (TTS)
# ---------------------------
LANG_VOICE_MAP = { "en": "en-US-JennyNeural", "ta": "ta-IN-ValluvarNeural", "hi": "hi-IN-MadhurNeural", "fr": "fr-FR-DeniseNeural", "es": "es-ES-ElviraNeural", "de": "de-DE-KatjaNeural" }

async def speak(text, lang="en"):
    voice = LANG_VOICE_MAP.get(lang, LANG_VOICE_MAP["en"])
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        file_path = fp.name
    try:
        communicate = edge_tts.Communicate(text, voice=voice)
        await communicate.save(file_path)
        playsound(file_path)
    except Exception as e:
        print(f"Error in TTS: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def start_asyncio_loop():
    global asyncio_loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    asyncio_loop = loop
    loop.run_forever()
asyncio_loop = None

# ---------------------------
# 5. Sentence Refinement, Translation, AND DOMAIN CHECK
# ---------------------------
ENGLISH_FILLER_WORDS = {"uh", "um", "like", "you know", "actually", "basically", "so", "i mean", "right", "well", "okay", "ok"}

def refine_english_sentence(text):
    words = text.lower().split()
    filtered_words = [word for word in words if word not in ENGLISH_FILLER_WORDS]
    refined_text = " ".join(filtered_words)
    return refined_text.capitalize()

def translate_to_english(text, source_lang):
    if source_lang == "en":
        return text
    translator = get_or_install_translator(source_lang, "en")
    if translator:
        return translator.translate(text)
    else:
        print(f"No translator found for {source_lang}. Cannot translate.")
        return text

def is_in_domain(text):
    """Checks if the text contains any of the required billing keywords."""
    text_lower = text.lower()
    for keyword in BILLING_KEYWORDS:
        if keyword in text_lower:
            print(f"[Domain Check] Found keyword: '{keyword}'")
            return True
    print("[Domain Check] No relevant keywords found.")
    return False

# ---------------------------
# 6. Main Processing Pipeline with Groq Whisper
# ---------------------------
def processing_pipeline(audio_data):
    print("\n[Processing] Transcribing with Groq Whisper...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        file_path = temp_file.name
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data)
    with open(file_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(file_path), file.read()),
            model="whisper-large-v3",
            response_format="json",
            language=args.language if args.language else None
        )
    original_text = transcription.text.strip()
    detected_language = getattr(transcription, 'language', 'en')
    os.remove(file_path)

    if not original_text:
        print("[Info] No speech detected in the audio.")
        return

    print(f"\n[Result] Original ({detected_language}): {original_text}")

    translated_text = translate_to_english(original_text, detected_language)
    if detected_language != "en":
        print(f"[Result] Translated (en): {translated_text}")

    refined_text = refine_english_sentence(translated_text)
    print(f"[Result] Refined (en): {refined_text}")

    # Check if the query is in-domain using keywords
    if is_in_domain(refined_text):
        bot_reply = f"Processing your billing query: {refined_text}"
    else:
        bot_reply = "Sorry, I can only answer questions related to billing."

    print(f"Bot: {bot_reply}")

    if asyncio_loop:
        asyncio.run_coroutine_threadsafe(speak(bot_reply, lang="en"), asyncio_loop)

# ---------------------------
# 7. Main Loop & Execution
# ---------------------------
def main_loop(args):
    silence_frames_needed = int(args.silence_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=silence_frames_needed)
    speaking = False
    speech_buffer = []

    print("\n Listening... (speak to activate)")
    while True:
        frame = q.get()
        is_speech = vad.is_speech(frame, samplerate)
        if not speaking:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.8 * ring_buffer.maxlen:
                speaking = True
                print(" Speaking detected...", end="", flush=True)
                for f, s in ring_buffer:
                    speech_buffer.append(f)
                ring_buffer.clear()
        else:
            speech_buffer.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                speaking = False
                print(" Silence detected. Processing...")
                audio_data = b"".join(speech_buffer)
                threading.Thread(target=processing_pipeline, args=(audio_data,)).start()
                speech_buffer = []
                ring_buffer.clear()
                print("\nListening... (speak to activate)")
if __name__ == "__main__":
    try:
        asyncio_thread = threading.Thread(target=start_asyncio_loop, daemon=True)
        asyncio_thread.start()
        time.sleep(0.5)  # Wait for asyncio loop to initialize
        main_thread = threading.Thread(target=main_loop, args=(args,), daemon=True)
        main_thread.start()
        with sd.RawInputStream(samplerate=samplerate, blocksize=frame_size, device=mic_index, channels=channels, dtype='int16', callback=audio_callback):
            print("Press Ctrl+C to exit.")
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"An error occurred: {e}")