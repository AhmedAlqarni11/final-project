import gradio as gr
from yt_fetch import download_audio, process_video  # Importing the split_audio function from yt_fetch.py
from stt_whisper import transcribe_multiple_audio_files  # Import the audio to text transcription function
from summarizer import summarize_text, translate_text
from agent import answer_question
from ingest import ingest_text_to_chroma

# Function to process video and text, and display video metadata
def process_video_and_text(url):
    audio_file_path = download_audio(url)  # Download the audio
    if "❌" in audio_file_path:
        return audio_file_path, ""  # Return error with an empty string for transcribed text

    # Check the audio length and split it if necessary (internally handled in process_video)
    audio_segments, full_audio_path = process_video(url)  # Using the updated process_video function
    if isinstance(audio_segments, str) and "❌" in audio_segments:
        return audio_segments, ""  # Return error with an empty string for transcribed text

    # Convert audio to text
    transcribed_text = transcribe_multiple_audio_files(audio_segments)  # Use parallel transcription
    if not transcribed_text:
        return "❌ Failed to extract text from audio.", ""  # If failed to extract text from audio

    # Store the transcribed text in the database (Chroma), without adding url in the text
    ingest_text_to_chroma(" ".join(transcribed_text))  # Pass the URL separately in the metadata

    # Return transcribed text and audio segments (the segments will be processed internally by process_video)
    return " ".join(transcribed_text), audio_segments  # audio_segments are handled by process_video

def summarize_and_return(text):
    summarized_text = summarize_text(text)
    ingest_text_to_chroma(summarized_text)
    return summarized_text

def translate_text_function(text, target_language="en"):
    translated_text = translate_text(text, target_language)
    ingest_text_to_chroma(translated_text)
    return translated_text

def answer_question_from_input(user_question, chat_history):
    answer = answer_question(user_question)
    chat_history.append(["user", user_question])
    chat_history.append(["assistant", answer])
    return chat_history

# Gradio Interface setup
with gr.Blocks() as interface:
    gr.Markdown("### 🎬 Video Content Extraction and Analysis")

    with gr.Tab("Extract Text & Summary"):
        with gr.Column():
            video_url = gr.Textbox(label="🔗 Enter video URL")
            process_video_btn = gr.Button("🔎 Extract text from video")
            full_text_output = gr.Textbox(label="📌 Extracted Text", interactive=False)
            
            # When the button is clicked, call process_video_and_text function
            process_video_btn.click(process_video_and_text, inputs=video_url, outputs=[full_text_output])

        # Add the summary section under the full text
        with gr.Column():
            summarize_btn = gr.Button("🔍 Show Summary")
            summarized_text_output = gr.Textbox(label="📑 Summary", interactive=False)
            summarize_btn.click(summarize_and_return, inputs=full_text_output, outputs=summarized_text_output)

    with gr.Tab("Translation"):
        with gr.Column():
            language_dropdown = gr.Dropdown(
                label="🌍 Choose Translation Language", 
                choices=["English", "Arabic", "French", "German", "Spanish", "Italian", "Portuguese", "Russian", "Japanese", "Chinese"],
                value="English", 
                interactive=True
            )
            translate_btn = gr.Button("🌍 Translate Text")
            translated_text_output = gr.Textbox(label="🌐 Translated Text", interactive=False)
            translate_btn.click(translate_text_function, inputs=[summarized_text_output, language_dropdown], outputs=translated_text_output)

    with gr.Tab("Q&A"):
        with gr.Column():
            user_question = gr.Textbox(label="❓ Enter Your Question")
            ask_question_btn = gr.Button("💡 Get Answer")
            chat_history = gr.Chatbot()
            chat_state = gr.State([])  # Keeps track of the conversation history
            ask_question_btn.click(answer_question_from_input, inputs=[user_question, chat_state], outputs=[chat_history])

    interface.css = """
        .gradio-chatbot .user { display: none; }
        .gradio-chatbot .assistant { display: none; }
    """
    interface.launch(share=True)
