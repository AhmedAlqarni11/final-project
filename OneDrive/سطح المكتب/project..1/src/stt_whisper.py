from concurrent.futures import ThreadPoolExecutor
import whisper
import re

# Load Whisper model of type 'small'
model = whisper.load_model("small")

def transcribe_audio(audio_path):
    """
    Convert audio to text using Whisper.
    """
    try:
        result = model.transcribe(audio_path)

        if "text" not in result:
            return f"❌ Failed to extract text from {audio_path}"

        text = result["text"]
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            return f"❌ Extracted text is empty from {audio_path}"

        return text  # Return only the extracted text

    except Exception as e:
        return f"❌ Error during audio-to-text conversion for {audio_path}: {e}"

def transcribe_multiple_audio_files(audio_paths):
    """
    Process multiple audio files in parallel to convert them into text using Whisper.
    """
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(transcribe_audio, audio_paths))  # Process multiple files in parallel
    return results

# Example usage
audio_paths = ["audio_1.wav", "audio_2.wav", "audio_3.wav"]  # List of audio files
transcriptions = transcribe_multiple_audio_files(audio_paths)

# Print all transcriptions
for transcription in transcriptions:
    print(transcription)
