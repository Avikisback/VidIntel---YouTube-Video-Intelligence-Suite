# 🎥 VidIntel - YouTube Video Intelligence Suite

VidIntel is a Streamlit-powered app that provides smart tools to extract and analyze insights from YouTube videos. It includes features like comment analysis, transcription, summarization, question-answering, translation, and keyword timestamp search.

---

## 🚀 Features

- **💬 YouTube Comments Analyzer**  
  Fetch comments from a YouTube video and analyze their sentiment using a RoBERTa model (positive, neutral, negative).

- **📝 Transcription Generator**  
  Download audio from a YouTube video and transcribe it using OpenAI’s Whisper model. The result is saved and viewable in the app.

- **🧠 Transcript Summarisation**  
  Summarize long transcripts into short, readable overviews using a pre-trained DistilBART summarizer.

- **❓ Ask Questions**  
  Ask any question based on a transcript. The app finds the most relevant context using sentence embeddings and answers using Flan-T5.

- **🌍 Translate Any Text**  
  Translate any given text to your desired language using the Google Translate API.

- **⏱️ Keyword Timestamp Search**  
  Search for a word in a transcript and view where it appears, along with exact timestamps.

---

## 🧰 Tech Stack

- Streamlit
- Hugging Face Transformers
- Sentence Transformers
- Whisper (OpenAI)
- yt-dlp
- googletrans
- YouTube Data API v3

---

## 🔮 Next Steps & Improvements

- **Better Q&A Bot**  
  Replace Flan-T5 with a stronger model (e.g., Mistral, GPT-4 API) for more accurate answers.

- **Smarter Summarisation**  
  Add options for TL;DR or bullet-point summaries. Fine-tune on spoken language data.

- **Multilingual Support**  
  Auto-detect and translate non-English transcripts for analysis.

- **Transcript Highlights**  
  Highlight key phrases using keyword extraction tools like KeyBERT.

- **Data Dashboard**  
  Add insights like comment trends, frequent keywords, and sentiment distribution.

- **Export Options**  
  Allow users to export transcripts, summaries, and answers in PDF or TXT.

