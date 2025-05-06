import os
import yt_dlp
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor

def download_audio(url, download_path="downloads/"):
    """
    Download only the audio from the URL and convert it to audio (WAV) with very low quality.
    """
    os.makedirs(download_path, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',  # Download the best audio available
        'outtmpl': os.path.join(download_path, '%(id)s.%(ext)s'),  # Output template
        'quiet': True,  # Suppress unnecessary output
        'noplaylist': True,  # Avoid downloading the entire playlist
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',  # Convert to WAV
            'preferredquality': '0',  # Use the lowest quality setting for fast processing
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info_dict = ydl.extract_info(url, download=True)
            video_id = info_dict.get('id', 'default_id')
            audio_file_path = os.path.join(download_path, f"{video_id}.wav")
            print(f"Audio downloaded successfully: {audio_file_path}")
            return audio_file_path
        except Exception as e:
            return f"❌ Error during video download: {str(e)}"

def split_audio(audio_path, segment_length_ms=1200000, download_path="downloads/"):
    """
    Split the audio file into 20-minute segments (1200000 ms).
    """
    if not os.path.exists(audio_path):
        return "❌ Audio file not found."

    try:
        audio = AudioSegment.from_file(audio_path, format="wav")
        print(f"Audio loaded from {audio_path}")
    except Exception as e:
        return f"❌ Error reading the audio file: {e}"

    segment_folder = os.path.join(download_path, os.path.basename(audio_path).split('.')[0])
    os.makedirs(segment_folder, exist_ok=True)
    print(f"Folder created: {segment_folder}")

    segments = []
    for i in range(0, len(audio), segment_length_ms):
        segment = audio[i:i + segment_length_ms]
        segment_file_path = os.path.join(segment_folder, f"segment_{i // segment_length_ms}.wav")
        segment.export(segment_file_path, format="wav", bitrate="16k")  # Export with very low bitrate
        segments.append(segment_file_path)

    print(f"Audio split into {len(segments)} segments")
    return segments

def process_video(url):
    """
    Full process: download audio, and split into segments if necessary (only if longer than 60 minutes).
    """
    audio_file_path = download_audio(url)
    
    if "❌" in audio_file_path:
        return audio_file_path  # Return error message if download failed

    # Check if audio duration is greater than 60 minutes before splitting
    audio = AudioSegment.from_file(audio_file_path, format="wav")
    duration_in_minutes = len(audio) / 60000  # Convert milliseconds to minutes

    # If audio duration is greater than 60 minutes, split it into 20-minute segments
    if duration_in_minutes > 60:
        print(f"Audio length is {duration_in_minutes} minutes, splitting into smaller segments.")
        segments = split_audio(audio_file_path, segment_length_ms=1200000)  # 20-minute segments
    else:
        print(f"Audio length is {duration_in_minutes} minutes, no splitting required.")
        segments = [audio_file_path]  # If the audio is less than 60 minutes, use it as is.

    return segments, audio_file_path

def process_multiple_videos(urls):
    """
    Process multiple videos in parallel.
    """
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_video, urls))
    return results


