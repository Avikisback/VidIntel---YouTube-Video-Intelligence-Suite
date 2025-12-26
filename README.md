# VidIntel - YouTube Video Intelligence Suite

VidIntel is a comprehensive Streamlit application designed to extract intelligence from YouTube videos. It offers a suite of tools for analyzing comments, transcribing audio, summarizing content, asking questions about the video, translating text, and searching for keywords within transcripts.

## Features

*   **💬 YouTube Comments Analyzer**: Fetch comments from a video and analyze their sentiment (Positive, Negative, Neutral).
*   **📝 Transcription Generator**: Download audio from a YouTube video and transcribe it using OpenAI's Whisper model.
*   **🧠 Transcript Summarisation**: Generate concise summaries of video transcripts using Hugging Face transformers.
*   **❓ Ask Questions**: Ask natural language questions about the video content and get answers based on the transcript.
*   **🌍 Translate Any Text**: Translate text between numerous languages.
*   **⏱️ Keyword Timestamp Search**: Search for specific keywords in a transcript and get their timestamps.

## Prerequisites

Before running the application, ensure you have the following installed:

1.  **Python 3.8+**
2.  **FFmpeg**: Required for audio processing.
    *   **Windows**: [Download FFmpeg](https://ffmpeg.org/download.html), extract it, and add the `bin` folder to your system PATH.
    *   **Mac**: `brew install ffmpeg`
    *   **Linux**: `sudo apt install ffmpeg`

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd vidintel
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

To use the **Comments Analyzer** feature, you need a YouTube Data API Key.

1.  Open `app.py`.
2.  Locate the line:
    ```python
    API_KEY = "YOUTUBE_API_KEY"
    ```
3.  Replace `"YOUTUBE_API_KEY"` with your actual API key from the [Google Cloud Console](https://console.cloud.google.com/).

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default web browser. Use the sidebar to navigate between different features.

## Notes

*   **Transcription**: The first time you run the transcription or summarization features, the application will download the necessary machine learning models. This may take some time depending on your internet connection.
*   **Performance**: Transcription and sentiment analysis can be resource-intensive. A machine with a GPU is recommended for faster processing, but it will work on a CPU as well.

## License

[MIT License](LICENSE)
