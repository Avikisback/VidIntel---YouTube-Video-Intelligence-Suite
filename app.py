import streamlit as st

st.set_page_config(page_title="VidIntel", layout="wide")

# === Sidebar navigation ===
st.sidebar.title("🎛️ VidIntel Tools")
option = st.sidebar.radio("Select a feature:", ["Comments", "Transcription", "Summarisation", "Ask Questions", "Translate Any Text", "Keyword Timestamp Search"])

st.title("🎥 VidIntel - YouTube Video Intelligence Suite")

# === Page content based on selection ===
if option == "Comments":
    st.header("💬 YouTube Comments Analyzer")
    st.write("Enter the video URL and number of comments to fetch.")

    import re
    import requests
    from transformers import pipeline

    # Function to extract video ID from URL
    def extract_video_id(youtube_url):
        pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
        return None

    # Inputs
    url = st.text_input("Enter YouTube Video URL")
    num_comments = st.number_input("Number of comments to fetch", min_value=100, max_value=1000, step=100, value=100)
    mode = st.radio("Choose comment output mode:", ["All Comments", "Sentiment Segmented Comments"])

    if url:
        video_id = extract_video_id(url)

        if video_id:
            st.success(f"Video ID: {video_id}")

            # YouTube API Setup
            API_KEY = "YOUTUBE_API_KEY"  
            YT_URL = "https://www.googleapis.com/youtube/v3/commentThreads"

            def get_comments(video_id, api_key, max_comments):
                comments = []
                next_page_token = None

                while len(comments) < max_comments:
                    params = {
                        'part': 'snippet',
                        'videoId': video_id,
                        'maxResults': 100,
                        'pageToken': next_page_token,
                        'key': api_key,
                        'textFormat': 'plainText'
                    }
                    response = requests.get(YT_URL, params=params)
                    data = response.json()

                    for item in data.get('items', []):
                        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                        comments.append(comment)

                    next_page_token = data.get('nextPageToken')
                    if not next_page_token:
                        break

                return comments[:max_comments]

            # Fetch and display
            comments = get_comments(video_id, API_KEY, int(num_comments))
            st.info(f"Total comments fetched: {len(comments)}")

            if mode == "All Comments":
                st.subheader("📃 All Comments")
                for i, comment in enumerate(comments, 1):
                    st.write(f"{i}. {comment}")

            elif mode == "Sentiment Segmented Comments":
                st.subheader("🧠 Sentiment-Based Comments")

                classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

                neg_l, neut_l, pos_l = [], [], []
                progress_bar = st.progress(0)
                for i, comment in enumerate(comments):
                    result = classifier(comment)
                    label = result[0]['label']
                    if label == 'LABEL_0':
                        neg_l.append(comment)
                    elif label == 'LABEL_2':
                        pos_l.append(comment)
                    else:
                        neut_l.append(comment)
                    progress_bar.progress((i + 1) / len(comments))

                st.success("✅ Sentiment analysis complete!")

                st.write("### 🔴 Negative Comments")
                for i, c in enumerate(neg_l, 1):
                    st.write(f"{i}. {c}")

                st.write("### 🟢 Positive Comments")
                for i, c in enumerate(pos_l, 1):
                    st.write(f"{i}. {c}")

                st.write("### 🟡 Neutral Comments")
                for i, c in enumerate(neut_l, 1):
                    st.write(f"{i}. {c}")
        else:
            st.error("❌ Invalid YouTube URL. Please try again.")

elif option == "Transcription":
    st.header("📝 Transcription Generator")
    st.write("Paste a YouTube video link and transcribe its audio using Whisper.")

    import yt_dlp
    import whisper
    import json
    import os

    # Function to download audio from YouTube
    def download_audio(url, output_file="audio"):
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_file,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    # Get input from user
    youtube_url = st.text_input("Enter YouTube video URL to transcribe")

    if st.button("Start Transcription") and youtube_url:
        try:
            with st.spinner("🔽 Downloading audio from YouTube..."):
                download_audio(youtube_url, "audio")
                st.success("✅ Audio downloaded successfully.")

            with st.spinner("🧠 Transcribing audio using Whisper..."):
                model = whisper.load_model("small")
                result = model.transcribe("audio.mp3")

                # Save transcript
                with open("transcription_result.json", "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                st.success("✅ Transcription complete and saved.")

                # Show transcript
                st.subheader("📄 Full Transcript")
                st.write(result["text"])

                # Optional: download JSON
                st.download_button(
                    "⬇️ Download transcript JSON",
                    data=json.dumps(result, indent=2),
                    file_name="transcription_result.json",
                    mime="application/json"
                )

                # 🔥 Delete temp audio file
                if os.path.exists("audio.mp3"):
                    os.remove("audio.mp3")
                    st.info("🗑️ Temporary audio file deleted.")
        except Exception as e:
            st.error(f"❌ Error: {e}")






elif option == "Summarisation":
    st.header("🧠 Transcript Summarisation")

    from transformers import pipeline
    default_text="Paste your transcription here"

    transcript_input = st.text_area(
        "Paste or edit the transcript here:",
        value=default_text,
        height=300
    )

    if st.button("Summarise") and transcript_input.strip():
        with st.spinner("🧠 Summarising..."):
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

            max_chunk_size = 1000
            transcript_chunks = [transcript_input[i:i + max_chunk_size]
                                 for i in range(0, len(transcript_input), max_chunk_size)]

            summary_chunks = []
            for chunk in transcript_chunks:
                input_length = len(chunk.split())
                min_len = max(10, int(input_length * 0.5))
                max_len = max(min_len + 5, int(input_length * 0.8))

                max_len = min(input_length, max_len)
                min_len = min(input_length - 1, min_len)

                summary = summarizer(
                    chunk,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False
                )
                summary_chunks.append(summary[0]['summary_text'])

            final_summary = "\n".join(summary_chunks)

        st.success("✅ Summary generated!")
        st.text_area("📋 Final Summary", value=final_summary, height=300)

elif option == "Ask Questions":
    st.header("❓ Ask Questions About the Video")

    from transformers import pipeline
    from sentence_transformers import SentenceTransformer
    import numpy as np

    # Load models
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    generator = pipeline("text2text-generation", model="google/flan-t5-base")

    # Input fields
    transcript = st.text_area("📄 Paste the transcript here", height=300)
    question = st.text_input("❓ Enter your question")

    # Helper: chunk text
    def chunk_text(text, chunk_size=50, overlap=10):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = words[i:i + chunk_size]
            if chunk:
                chunks.append(" ".join(chunk))
        return chunks

    # Cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Ask with Flan-T5
    def ask_flan(question, context):
        prompt = f"""
You are a helpful assistant that answers questions based on the given context. 
Provide a detailed, well-structured answer in full sentences. 
If the question asks for multiple points, list them clearly and concisely.

Context:
{context}

Question:
{question}

Answer:
"""
        response = generator(prompt, max_new_tokens=300, do_sample=False)
        return response[0]['generated_text'].strip()

    # Button logic
    if st.button("Answer") and transcript.strip() and question.strip():
        with st.spinner("🧠 Thinking..."):
            chunks = chunk_text(transcript)
            chunk_embeddings = embedding_model.encode(chunks)
            query_embedding = embedding_model.encode([question])[0]

            similarities = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]
            top_indices = np.argsort(similarities)[::-1][:3]  # Use top 3
            context = "\n".join([chunks[i] for i in top_indices])

            answer = ask_flan(question, context)

        st.subheader("💬 Answer")
        st.write(answer)

elif option == "Translate Any Text":
    st.header("🌍 Translate Any Text")

    from googletrans import Translator

    LANGUAGES = {
        'af': 'afrikaans', 'sq': 'albanian', 'am': 'amharic', 'ar': 'arabic', 'hy': 'armenian',
        'az': 'azerbaijani', 'eu': 'basque', 'be': 'belarusian', 'bn': 'bengali', 'bs': 'bosnian',
        'bg': 'bulgarian', 'ca': 'catalan', 'ceb': 'cebuano', 'ny': 'chichewa',
        'zh-cn': 'chinese (simplified)', 'zh-tw': 'chinese (traditional)', 'co': 'corsican',
        'hr': 'croatian', 'cs': 'czech', 'da': 'danish', 'nl': 'dutch', 'en': 'english',
        'eo': 'esperanto', 'et': 'estonian', 'tl': 'filipino', 'fi': 'finnish', 'fr': 'french',
        'fy': 'frisian', 'gl': 'galician', 'ka': 'georgian', 'de': 'german', 'el': 'greek',
        'gu': 'gujarati', 'ht': 'haitian creole', 'ha': 'hausa', 'haw': 'hawaiian',
        'he': 'hebrew', 'hi': 'hindi', 'hmn': 'hmong', 'hu': 'hungarian', 'is': 'icelandic',
        'ig': 'igbo', 'id': 'indonesian', 'ga': 'irish', 'it': 'italian', 'ja': 'japanese',
        'jw': 'javanese', 'kn': 'kannada', 'kk': 'kazakh', 'km': 'khmer', 'ko': 'korean',
        'ku': 'kurdish', 'ky': 'kyrgyz', 'lo': 'lao', 'la': 'latin', 'lv': 'latvian',
        'lt': 'lithuanian', 'lb': 'luxembourgish', 'mk': 'macedonian', 'mg': 'malagasy',
        'ms': 'malay', 'ml': 'malayalam', 'mt': 'maltese', 'mi': 'maori', 'mr': 'marathi',
        'mn': 'mongolian', 'my': 'myanmar', 'ne': 'nepali', 'no': 'norwegian', 'or': 'odia',
        'ps': 'pashto', 'fa': 'persian', 'pl': 'polish', 'pt': 'portuguese', 'pa': 'punjabi',
        'ro': 'romanian', 'ru': 'russian', 'sm': 'samoan', 'gd': 'scots gaelic', 'sr': 'serbian',
        'st': 'sesotho', 'sn': 'shona', 'sd': 'sindhi', 'si': 'sinhala', 'sk': 'slovak',
        'sl': 'slovenian', 'so': 'somali', 'es': 'spanish', 'su': 'sundanese', 'sw': 'swahili',
        'sv': 'swedish', 'tg': 'tajik', 'ta': 'tamil', 'te': 'telugu', 'th': 'thai',
        'tr': 'turkish', 'uk': 'ukrainian', 'ur': 'urdu', 'ug': 'uyghur', 'uz': 'uzbek',
        'vi': 'vietnamese', 'cy': 'welsh', 'xh': 'xhosa', 'yi': 'yiddish', 'yo': 'yoruba', 'zu': 'zulu'
    }

    input_text = st.text_area("Enter text to translate", height=200)
    target_lang = st.text_input("Enter target language (e.g., spanish, hindi)").lower().strip()

    if target_lang == "chinese":
        chinese_style = st.radio("Choose Chinese style", ["Traditional", "Simplified"])
        target_lang = f"chinese ({chinese_style.lower()})"

    if st.button("Translate"):
        if target_lang in LANGUAGES.values():
            lang_code = [k for k, v in LANGUAGES.items() if v == target_lang][0]
            try:
                translator = Translator()
                translated = translator.translate(input_text, dest=lang_code)
                st.subheader("🗣️ Translated Output")
                st.text_area("Translation:", value=translated.text, height=300)
            except Exception as e:
                st.error(f"Translation failed: {e}")
        else:
            st.error("❌ Invalid language. Please try again.")
elif option == "Keyword Timestamp Search":
    st.header("⏱️ Search Transcript for Word Timestamps")

    import json
    from datetime import timedelta

    def format_time(seconds):
        return str(timedelta(seconds=int(seconds)))

    uploaded_file = st.file_uploader("📤 Upload the transcript JSON file", type="json")

    if uploaded_file:
        result = json.load(uploaded_file)
        segments = result.get("segments", [])

        keyword = st.text_input("🔍 Enter a keyword to search for").strip().lower()

        if keyword:
            matches = []
            for seg in segments:
                if keyword in seg['text'].lower():
                    matches.append((format_time(seg['start']), seg['text']))

            if matches:
                st.success(f"✅ Found {len(matches)} match(es) for '{keyword}':")
                for timestamp, text in matches:
                    st.markdown(f"**{timestamp}** — {text}")
            else:
                st.warning(f"No matches found for '{keyword}'.")
