#!/usr/bin/env python3
import os
import datetime
import openai
import yt_dlp
import requests
from dotenv import load_dotenv
from fpdf import FPDF
import gradio as gr
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import base64
import mimetypes

# Load environment variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file.")

### HELPER FUNCTIONS ###

def transcribe_audio(audio_file_path: str) -> str:
    """Transcribes an audio file using OpenAI's Whisper API."""
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript["text"]
    except Exception as e:
        return f"Error during transcription: {e}"

def generate_output(text: str, task: str) -> str:
    """
    Uses GPT-4 to process text for a given task:
      - "lyrics": rewrite lyrics with proper formatting.
      - "article": generate an article.
      - "summarize": produce a summary.
    """
    if task == "lyrics":
        prompt = f"""You are a digital music consultant.
Given the following transcription of a song, please rewrite the lyrics exactly as sung, with clear line breaks for verses.
Transcription:
{text}"""
    elif task == "article":
        prompt = f"""You are a digital music consultant.
Given the following transcription of an audio file or website text, generate a well-structured article discussing its content.
Transcription:
{text}"""
    elif task == "summarize":
        prompt = f"""You are a digital music consultant.
Please summarize the following text in a concise and clear manner:
{text}"""
    else:
        prompt = text

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error during GPT processing: {e}"

def generate_press_release(song_title: str, artist_name: str, release_date: str, album_description: str) -> str:
    """Generates a professional press release using GPT-4."""
    prompt = f"""You are a digital music consultant and marketing expert.
Generate a professional press release for the following details:
Song/Album Title: {song_title}
Artist Name: {artist_name}
Release Date: {release_date}
Album Description: {album_description}
Output only the press release text."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating press release: {e}"

def generate_social_media_post(song_title: str, artist_name: str) -> str:
    """Generates a creative social media post using GPT-4."""
    prompt = f"""You are a digital music consultant with expertise in social media marketing.
Write an engaging and concise social media post to promote the song "{song_title}" by {artist_name}.
Output only the post text."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating social media post: {e}"

def generate_epk(artist_name: str, background_info: str, achievements: str, social_links: str, press_quotes: str) -> str:
    """Generates a comprehensive Electronic Press Kit (EPK) using GPT-4."""
    prompt = f"""You are a digital music consultant and marketing expert.
Generate a comprehensive Electronic Press Kit (EPK) for the artist "{artist_name}".
Include:
- Background Information: {background_info}
- Achievements: {achievements}
- Social Media/Website Links: {social_links}
- Press Quotes: {press_quotes}
Output only the EPK text."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating EPK: {e}"

def text_to_pdf(text: str, output_filename: str) -> str:
    """Generates a PDF file from the provided text and returns the filename."""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, text)
        pdf.output(output_filename)
        return output_filename
    except Exception as e:
        return f"Error generating PDF: {e}"

def create_epk_pdf(epk_text: str, output_filename: str, photos: list = None) -> str:
    """
    Creates a PDF for the EPK that includes the generated EPK text.
    If photos are provided, they are embedded on a new page.
    (Video links are appended to the text in the callback.)
    """
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, epk_text)
        if photos:
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, "Photos:", ln=True)
            for photo in photos:
                photo_path = photo.name if hasattr(photo, "name") else str(photo)
                try:
                    pdf.image(photo_path, w=100)
                    pdf.ln(10)
                except Exception as e:
                    pdf.cell(0, 10, f"Error adding photo {photo_path}: {e}", ln=True)
        pdf.output(output_filename)
        return output_filename
    except Exception as e:
        return f"Error creating EPK PDF: {e}"

def get_audio(url: str, desired_format: str = "mp3") -> str:
    """
    Uses yt-dlp to download audio from a URL and convert it to the desired format.
    Returns the full file path of the downloaded audio.
    """
    downloads_dir = os.path.join(os.getcwd(), "downloads")
    os.makedirs(downloads_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_template = os.path.join(downloads_dir, f"{timestamp}_%(title)s.%(ext)s")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': desired_format,
            'preferredquality': '192',
        }],
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_filename = ydl.prepare_filename(info)
            base, ext = os.path.splitext(downloaded_filename)
            if ext.lower() != f".{desired_format}":
                downloaded_filename = base + f".{desired_format}"
            return downloaded_filename
    except Exception as e:
        return f"Error downloading audio: {e}"

def chat_with_api(prompt: str) -> str:
    """A simple wrapper to interact with GPT-4."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {e}"

###############################################
# NEW HELPER FUNCTION: Create PDF with Images
###############################################
def create_pdf_with_images(text: str, output_filename: str, image_paths: list) -> str:
    """
    Creates a PDF that includes the provided text on the first page,
    and embeds each image (from image_paths) on subsequent pages.
    """
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, text)
        for img in image_paths:
            pdf.add_page()
            try:
                pdf.image(img, w=100)
            except Exception as e:
                pdf.cell(0, 10, f"Error embedding image {img}: {e}", ln=True)
        pdf.output(output_filename)
        return output_filename
    except Exception as e:
        return f"Error generating PDF with images: {e}"

###############################################
# NEW FUNCTION: Crawl Website and Generate Article PDF with Optional Images
###############################################
def crawl_and_generate_article(url: str, pdf_filename: str, include_images: bool = False) -> tuple[str, str]:
    """
    Crawls the given website URL using BeautifulSoup, extracts its primary textual content,
    and then uses GPT-4 to generate a formal, well-structured article based on that content.
    Optionally, if include_images is True, it downloads image URLs from the page and
    uses create_pdf_with_images to embed them in the resulting PDF.
    
    Returns:
      A tuple (article_text, pdf_file_path).
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        return (f"Error retrieving the website: {e}", None)
    
    try:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove script and style elements
        for tag in soup(["script", "style"]):
            tag.decompose()
        raw_text = soup.get_text(separator="\n")
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        cleaned_text = "\n".join(lines)
        if len(cleaned_text) > 10000:
            cleaned_text = cleaned_text[:10000]
    except Exception as e:
        return (f"Error parsing website content: {e}", None)
    
    article_text = generate_output(cleaned_text, task="article")
    
    if include_images:
        image_tags = soup.find_all("img")
        image_urls = []
        for tag in image_tags:
            src = tag.get("src")
            if src:
                if not src.startswith("http"):
                    src = urljoin(url, src)
                image_urls.append(src)
        downloaded_images = []
        downloads_dir = os.path.join(os.getcwd(), "downloads")
        os.makedirs(downloads_dir, exist_ok=True)
        for img_url in image_urls:
            try:
                img_response = requests.get(img_url, stream=True, timeout=10)
                img_response.raise_for_status()
                img_filename = os.path.join(downloads_dir, f"img_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg")
                with open(img_filename, "wb") as f:
                    for chunk in img_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                downloaded_images.append(img_filename)
            except Exception as e:
                print(f"Error downloading image {img_url}: {e}")
        pdf_path = create_pdf_with_images(article_text, pdf_filename, downloaded_images)
    else:
        pdf_path = text_to_pdf(article_text, pdf_filename)
    
    return (article_text, pdf_path)

###############################################
# NEW FUNCTION: Crawl Website for Images Only
###############################################
def crawl_images_gui(crawl_url: str) -> list:
    """
    Crawls the given website URL using BeautifulSoup, extracts all image URLs,
    downloads them, and returns a list of local file paths to the downloaded images.
    """
    try:
        response = requests.get(crawl_url)
        response.raise_for_status()
    except Exception as e:
        return [f"Error retrieving website: {e}"]
    
    try:
        soup = BeautifulSoup(response.text, 'html.parser')
        image_tags = soup.find_all("img")
        image_urls = []
        for tag in image_tags:
            src = tag.get("src")
            if src:
                if not src.startswith("http"):
                    src = urljoin(crawl_url, src)
                image_urls.append(src)
    except Exception as e:
        return [f"Error parsing website content: {e}"]

    downloaded_images = []
    downloads_dir = os.path.join(os.getcwd(), "downloads")
    os.makedirs(downloads_dir, exist_ok=True)
    for img_url in image_urls:
        try:
            img_response = requests.get(img_url, stream=True, timeout=10)
            img_response.raise_for_status()
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
            img_filename = os.path.join(downloads_dir, f"webimg_{timestamp}.jpg")
            with open(img_filename, "wb") as f:
                for chunk in img_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            downloaded_images.append(img_filename)
        except Exception as e:
            print(f"Error downloading image {img_url}: {e}")
    return downloaded_images

###############################################
# CALLBACK: Download Chat Conversation as PDF
###############################################
def download_chat_pdf(history, pdf_filename):
    """Converts the chat conversation history into a PDF file."""
    if not pdf_filename:
        pdf_filename = "chat_conversation.pdf"
    conversation_text = ""
    for msg in history:
        conversation_text += f"{msg['role'].capitalize()}: {msg['content']}\n\n"
    return text_to_pdf(conversation_text, pdf_filename)

###############################################
# CALLBACK: Update Visible Groups Based on Dropdown Selection
###############################################
def update_function_view(choice: str):
    chat_vis = True if choice == "Chat" else False
    transcribe_vis = True if choice == "Transcribe Audio" else False
    lyrics_vis = True if choice == "Generate Lyrics (PDF)" else False
    article_vis = True if choice == "Generate Article (PDF)" else False
    summarize_vis = True if choice == "Summarize Text (PDF)" else False
    getaudio_vis = True if choice == "Get Audio" else False
    pressrelease_vis = True if choice == "Press Release (PDF)" else False
    social_vis = True if choice == "Social Media Post" else False
    epk_vis = True if choice == "Generate EPK (PDF)" else False
    crawl_vis = True if choice == "Generate Website Article (PDF)" else False
    web_images_vis = True if choice == "Get Web Images" else False
    return (
        gr.update(visible=chat_vis),
        gr.update(visible=transcribe_vis),
        gr.update(visible=lyrics_vis),
        gr.update(visible=article_vis),
        gr.update(visible=summarize_vis),
        gr.update(visible=getaudio_vis),
        gr.update(visible=pressrelease_vis),
        gr.update(visible=social_vis),
        gr.update(visible=epk_vis),
        gr.update(visible=crawl_vis),
        gr.update(visible=web_images_vis)
    )

###############################################
# GRADIO GUI SETUP
###############################################
with gr.Blocks(title="ThisIsMusic.ai - Digital Music Consultant") as demo:
    gr.Markdown("## ThisIsMusic.ai\nYour digital music consultant for music production, marketing, and creative functions.")
    
    # Dropdown for function selection (including new options)
    function_choice = gr.Dropdown(
        label="Select Function",
        choices=[
            "Chat",
            "Transcribe Audio",
            "Generate Lyrics (PDF)",
            "Generate Article (PDF)",
            "Summarize Text (PDF)",
            "Get Audio",
            "Press Release (PDF)",
            "Social Media Post",
            "Generate EPK (PDF)",
            "Generate Website Article (PDF)",
            "Get Web Images"
        ],
        value="Chat",
        info="Select a function to perform. The corresponding input fields will appear below."
    )
    
    # Groups for each function – only one visible at a time.
    with gr.Group(visible=True) as chat_group:
        # Chat history appears at the top.
        chat_output = gr.Chatbot(label="Conversation", type="messages")
        # Input field and send button.
        chat_input = gr.Textbox(label="Your Message", placeholder="Type your message here...", lines=2)
        # NEW: Optional image upload for chat (returns binary data).
        chat_image_input = gr.File(label="Upload Image (optional)", type="binary", file_types=["image"])
        chat_button = gr.Button("Send")
        # Components for downloading the chat conversation.
        chat_pdf_filename = gr.Textbox(label="Chat PDF Filename", placeholder="chat_conversation.pdf")
        download_chat_button = gr.Button("Download Chat as PDF")
        chat_pdf_file = gr.File(label="Download Chat PDF")
    
    with gr.Group(visible=False) as transcribe_group:
        audio_file_input_trans = gr.File(label="Upload Audio File")
        audio_url_input_trans = gr.Textbox(label="Or Enter Audio URL", placeholder="Enter URL...")
        format_radio_trans = gr.Radio(["mp3", "wav"], label="Audio Format", value="mp3")
        transcribe_run = gr.Button("Run Transcription")
        transcribe_result = gr.Textbox(label="Transcription Output", interactive=False)
    
    with gr.Group(visible=False) as lyrics_group:
        audio_file_input_lyrics = gr.File(label="Upload Audio File")
        audio_url_input_lyrics = gr.Textbox(label="Or Enter Audio URL", placeholder="Enter URL...")
        format_radio_lyrics = gr.Radio(["mp3", "wav"], label="Audio Format", value="mp3")
        lyrics_pdf_name = gr.Textbox(label="Output PDF Filename", placeholder="lyrics.pdf")
        lyrics_run = gr.Button("Generate Lyrics")
        lyrics_output = gr.Textbox(label="Lyrics Output", interactive=False)
    
    with gr.Group(visible=False) as article_group:
        audio_file_input_article = gr.File(label="Upload Audio File")
        audio_url_input_article = gr.Textbox(label="Or Enter Audio URL", placeholder="Enter URL...")
        format_radio_article = gr.Radio(["mp3", "wav"], label="Audio Format", value="mp3")
        article_pdf_name = gr.Textbox(label="Output PDF Filename", placeholder="article.pdf")
        article_run = gr.Button("Generate Article")
        article_output = gr.Textbox(label="Article Output", interactive=False)
    
    with gr.Group(visible=False) as summarize_group:
        text_file_input = gr.File(label="Upload Text File")
        text_url_input = gr.Textbox(label="Or Enter Text URL", placeholder="Enter URL...")
        summarize_pdf_name = gr.Textbox(label="Output PDF Filename", placeholder="summary.pdf")
        summarize_run = gr.Button("Summarize Text")
        summarize_output = gr.Textbox(label="Summary Output", interactive=False)
    
    with gr.Group(visible=False) as getaudio_group:
        audio_url_input_get = gr.Textbox(label="Enter Audio URL", placeholder="Enter URL...")
        format_radio_get = gr.Radio(["mp3", "wav"], label="Audio Format", value="mp3")
        getaudio_run = gr.Button("Get Audio")
        download_audio_file = gr.File(label="Downloaded Audio File")
    
    with gr.Group(visible=False) as pressrelease_group:
        pr_song_title = gr.Textbox(label="Song/Album Title", placeholder="Enter title...")
        pr_artist_name = gr.Textbox(label="Artist Name", placeholder="Enter artist name...")
        pr_release_date = gr.Textbox(label="Release Date (YYYY-MM-DD)", placeholder="2025-01-01")
        pr_description = gr.Textbox(label="Album Description", placeholder="Enter description...", lines=3)
        pr_pdf_name = gr.Textbox(label="Press Release PDF Filename", placeholder="press_release.pdf")
        pr_run = gr.Button("Generate Press Release")
        pr_output = gr.Textbox(label="Press Release Output", interactive=False)
    
    with gr.Group(visible=False) as social_group:
        sp_song_title = gr.Textbox(label="Song Title", placeholder="Enter song title...")
        sp_artist_name = gr.Textbox(label="Artist Name", placeholder="Enter artist name...")
        social_run = gr.Button("Generate Social Media Post")
        social_output = gr.Textbox(label="Social Media Post Output", interactive=False)
    
    with gr.Group(visible=False) as epk_group:
        epk_artist_name = gr.Textbox(label="Artist Name", placeholder="Enter artist name...")
        epk_background = gr.Textbox(label="Background Information", placeholder="Enter background info...", lines=3)
        epk_achievements = gr.Textbox(label="Achievements", placeholder="Enter achievements...", lines=3)
        epk_social_links = gr.Textbox(label="Social/Website Links", placeholder="Enter links, separated by commas...")
        epk_press_quotes = gr.Textbox(label="Press Quotes", placeholder="Enter press quotes...", lines=2)
        epk_video_links = gr.Textbox(label="Video Links", placeholder="Enter video links, separated by commas...", value="")
        epk_pdf_name = gr.Textbox(label="EPK PDF Filename", placeholder="epk.pdf")
        # UPDATED: Use type "binary" with an accept filter for images.
        epk_photos = gr.File(label="Upload Photos", file_count="multiple", type="binary", file_types=["image"])
        epk_run = gr.Button("Generate EPK")
        epk_output = gr.Textbox(label="EPK Output", interactive=False)
    
    with gr.Group(visible=False) as crawl_group:
        crawl_url = gr.Textbox(label="Website URL", placeholder="Enter website URL to crawl...")
        crawl_pdf_name = gr.Textbox(label="Output PDF Filename", placeholder="article.pdf")
        crawl_include_images = gr.Radio(choices=["Yes", "No"], label="Include Images?", value="No")
        crawl_uploaded_images = gr.File(label="Upload Additional Images", file_count="multiple", type="binary", file_types=["image"])
        crawl_run = gr.Button("Generate Website Article")
        crawl_output = gr.Textbox(label="Article Output", interactive=False)
        crawl_pdf_file = gr.File(label="Download PDF")
    
    with gr.Group(visible=False) as images_group:
        images_url = gr.Textbox(label="Website URL", placeholder="Enter website URL for images...")
        images_run = gr.Button("Get Web Images")
        images_gallery = gr.Gallery(label="Web Images", show_label=True)
    
    # Shared output for generated PDFs (if needed)
    pdf_file = gr.File(label="Download PDF")
    
    ### CALLBACK FUNCTIONS ###
    def send_chat(message, image, history):
        # If an image is provided, encode it and include it in the message payload.
        if image is not None:
            if isinstance(image, dict) and "data" in image:
                file_data = image["data"]
                file_name = image.get("name", "uploaded_image")
            else:
                file_data = image
                file_name = "uploaded_image"
            mime_type, _ = mimetypes.guess_type(file_name)
            if not mime_type:
                mime_type = "application/octet-stream"
            encoded_data = base64.b64encode(file_data).decode("utf-8")
            image_str = f"data:{mime_type};base64,{encoded_data}"
            message_content = [{"type": "text", "text": message}, {"type": "image_url", "image_url": {"url": image_str}}]
        else:
            message_content = message
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-0613",
                messages=[{"role": "user", "content": message_content}],
                max_tokens=150,
                temperature=0.7,
            )
            reply = response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            reply = f"Error: {e}"
        history = history + [{"role": "user", "content": message_content}, {"role": "assistant", "content": reply}]
        return "", history

    def transcribe_audio_gui(file, url, format_choice):
        if file is not None:
            file_path = file.name
        elif url:
            if "youtube.com" in url.lower() or "youtu.be" in url.lower():
                return "YouTube links are not supported. Please use SoundCloud or another provider."
            file_path = get_audio(url, desired_format=format_choice)
        else:
            return "No audio provided."
        transcript = transcribe_audio(file_path)
        return transcript

    def generate_lyrics_gui(file, url, format_choice, pdf_filename):
        transcript = transcribe_audio_gui(file, url, format_choice)
        if transcript.startswith("Error") or transcript == "No audio provided.":
            return transcript, None
        lyrics = generate_output(transcript, task="lyrics")
        pdf_path = text_to_pdf(lyrics, pdf_filename)
        return lyrics, pdf_path

    def generate_article_gui(file, url, format_choice, pdf_filename):
        transcript = transcribe_audio_gui(file, url, format_choice)
        if transcript.startswith("Error") or transcript == "No audio provided.":
            return transcript, None
        article = generate_output(transcript, task="article")
        pdf_path = text_to_pdf(article, pdf_filename)
        return article, pdf_path

    def summarize_text_gui(file, url, pdf_filename):
        if file is not None:
            text = file.read().decode("utf-8")
        elif url:
            try:
                r = requests.get(url)
                r.raise_for_status()
                text = r.text
            except Exception as e:
                return f"Error downloading text: {e}", None
        else:
            return "No text provided.", None
        summary = generate_output(text, task="summarize")
        pdf_path = text_to_pdf(summary, pdf_filename)
        return summary, pdf_path

    def get_audio_gui(url, format_choice):
        if "youtube.com" in url.lower() or "youtu.be" in url.lower():
            return "YouTube links are not supported. Please use SoundCloud or another provider."
        file_path = get_audio(url, desired_format=format_choice)
        return file_path

    def press_release_gui(song_title, artist_name, release_date, description, pdf_filename):
        pr = generate_press_release(song_title, artist_name, release_date, description)
        pdf_path = text_to_pdf(pr, pdf_filename)
        return pr, pdf_path

    def social_post_gui(song_title, artist_name):
        post = generate_social_media_post(song_title, artist_name)
        return post

    def epk_gui(artist_name, background, achievements, social_links, press_quotes, video_links, pdf_filename, photos):
        epk_text = generate_epk(artist_name, background, achievements, social_links, press_quotes)
        if video_links.strip():
            epk_text += "\n\nVideo Links: " + video_links
        pdf_path = create_epk_pdf(epk_text, pdf_filename, photos)
        return epk_text, pdf_path

    def crawl_article_gui(crawl_url, pdf_filename, include_images_choice, uploaded_images):
        include_images = True if include_images_choice.lower() == "yes" else False
        article_text, pdf_path = crawl_and_generate_article(crawl_url, pdf_filename, include_images=include_images)
        if uploaded_images:
            pdf_path = create_pdf_with_images(article_text, pdf_filename, uploaded_images)
        return article_text, pdf_path

    def crawl_images_callback(crawl_url):
        images = crawl_images_gui(crawl_url)
        return images

    def download_chat_pdf_callback(chat_history, pdf_filename):
        return download_chat_pdf(chat_history, pdf_filename)

    ### LINKING THE DROPDOWN TO UPDATE VISIBLE GROUPS ###
    function_choice.change(
        update_function_view,
        inputs=function_choice,
        outputs=[chat_group, transcribe_group, lyrics_group, article_group, summarize_group, getaudio_group, pressrelease_group, social_group, epk_group, crawl_group, images_group]
    )
    
    ### LINKING COMPONENT ACTIONS ###
    chat_button.click(send_chat, inputs=[chat_input, chat_image_input, chat_output], outputs=[chat_input, chat_output])
    transcribe_run.click(transcribe_audio_gui, inputs=[audio_file_input_trans, audio_url_input_trans, format_radio_trans], outputs=transcribe_result)
    lyrics_run.click(generate_lyrics_gui, inputs=[audio_file_input_lyrics, audio_url_input_lyrics, format_radio_lyrics, lyrics_pdf_name], outputs=[lyrics_output, pdf_file])
    article_run.click(generate_article_gui, inputs=[audio_file_input_article, audio_url_input_article, format_radio_article, article_pdf_name], outputs=[article_output, pdf_file])
    summarize_run.click(summarize_text_gui, inputs=[text_file_input, text_url_input, summarize_pdf_name], outputs=[summarize_output, pdf_file])
    getaudio_run.click(get_audio_gui, inputs=[audio_url_input_get, format_radio_get], outputs=download_audio_file)
    pr_run.click(press_release_gui, inputs=[pr_song_title, pr_artist_name, pr_release_date, pr_description, pr_pdf_name], outputs=[pr_output, pdf_file])
    social_run.click(social_post_gui, inputs=[sp_song_title, sp_artist_name], outputs=social_output)
    epk_run.click(epk_gui, inputs=[epk_artist_name, epk_background, epk_achievements, epk_social_links, epk_press_quotes, epk_video_links, epk_pdf_name, epk_photos], outputs=[epk_output, pdf_file])
    crawl_run.click(crawl_article_gui, inputs=[crawl_url, crawl_pdf_name, crawl_include_images, crawl_uploaded_images], outputs=[crawl_output, crawl_pdf_file])
    images_run.click(crawl_images_callback, inputs=[images_url], outputs=[images_gallery])
    download_chat_button.click(download_chat_pdf_callback, inputs=[chat_output, chat_pdf_filename], outputs=chat_pdf_file)
    
    demo.launch(share=True, server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
    
if __name__ == "__main__":
    # This forces using the Heroku-assigned port if available.
    port = int(os.environ.get("PORT", 7860))
    print("Binding to port:", port)
    demo.launch(server_name="0.0.0.0", server_port=port)