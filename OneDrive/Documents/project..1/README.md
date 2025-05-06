# Video-to-Text Extraction and Summarization Project

https://drive.google.com/drive/folders/1XS33rpd8nu4n00-9JFJZx_4OHQJtH68X

## Introduction
The **Video-to-Text Extraction and Summarization** project aims to download videos from **YouTube**, extract audio from them, and convert that audio to text using the **Whisper** model. Then, the extracted text is summarized using **OpenAI GPT-4**. The project also includes splitting the audio into one-minute segments to manage content more effectively.

## Requirements

The following libraries are required to run this project:

- **langchain**: For managing models and data retrieval.
- **gradio**: To build the interactive user interface.
- **openai**: To access the **OpenAI GPT-4** API.
- **yt-dlp**: To download videos from **YouTube**.
- **whisper**: To convert audio to text using the **Whisper** model.
- **pinecone-client**: For vector database (Pinecone).
- **chromadb**: For storing and retrieving data with **Chroma**.
- **langsmith**: For evaluating model performance.
- **python-dotenv**: For loading environment variables from `.env` files.
- **requests**: For making HTTP requests.
- **pandas**: For data analysis.
- **scipy**: For scientific and mathematical operations.
- **pytest**: For unit testing.

### Installation
You can install all the required libraries using **`requirements.txt`** with the following command:

```bash
pip install -r requirements.txt
