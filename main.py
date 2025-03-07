#!/usr/bin/env python3
import os
import datetime
import openai
import yt_dlp
import requests
from dotenv import load_dotenv
from fpdf import FPDF

# Load environment variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in the .env file.")

### FUNCTION DEFINITIONS ###

def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcribes the given audio file using OpenAI's Whisper API.
    Expects a valid audio file path (e.g., .mp3, .wav).
    """
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript["text"]
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""

def generate_output(text: str, task: str) -> str:
    """
    Uses the GPT-4 API to process text based on the specified task.
    task: "lyrics" (rewrite as formatted song lyrics),
          "article" (create a formatted article),
          "summarize" (produce a summary of a text).
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
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during GPT processing: {e}")
        return ""

def generate_press_release(song_title: str, artist_name: str, release_date: str, album_description: str) -> str:
    """
    Generates a professional press release for an upcoming song or album release.
    """
    prompt = f"""You are a digital music consultant and marketing expert.
Generate a professional press release for the following release details:
Song/Album Title: {song_title}
Artist Name: {artist_name}
Release Date: {release_date}
Album Description: {album_description}

The press release should be engaging, informative, and formatted for media distribution. Output only the press release text.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating press release: {e}")
        return ""

def generate_social_media_post(song_title: str, artist_name: str) -> str:
    """
    Generates a creative social media post to promote a song.
    """
    prompt = f"""You are a digital music consultant with expertise in social media marketing.
Write an engaging and concise social media post to promote the song "{song_title}" by {artist_name}.
Include a call-to-action and a catchy tone suitable for platforms like Instagram or Twitter.
Output only the post text.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating social media post: {e}")
        return ""

def generate_epk(artist_name: str, background_info: str, achievements: str, social_links: str, press_quotes: str) -> str:
    """
    Generates a comprehensive Electronic Press Kit (EPK) for an artist.
    """
    prompt = f"""You are a digital music consultant and marketing expert.
Generate a comprehensive Electronic Press Kit (EPK) for the artist "{artist_name}".
Include the following details:
- Background Information: {background_info}
- Achievements: {achievements}
- Social Media/Website Links: {social_links}
"""
    if press_quotes:
        prompt += f"- Press Quotes: {press_quotes}\n"
    prompt += "\nThe EPK should be engaging, professional, and suitable for media and industry professionals. Output only the EPK text."
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating EPK: {e}")
        return ""

def text_to_pdf(text: str, output_filename: str) -> str:
    """
    Generates a PDF file from the provided text using fpdf and returns the filename.
    """
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, text)
        pdf.output(output_filename)
        print(f"PDF successfully saved as {output_filename}")
        return output_filename
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return f"Error generating PDF: {e}"

def create_epk_pdf(epk_text: str, output_filename: str, photos: list = None) -> str:
    """
    Creates a PDF for the EPK that includes the generated EPK text.
    If photos are provided, they are embedded on a new page.
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
        print(f"Error creating EPK PDF: {e}")
        return f"Error creating EPK PDF: {e}"

def get_audio(url: str, desired_format: str = "mp3") -> str:
    """
    Uses yt-dlp to download audio from a given URL and convert it to the desired format.
    Returns the full file path of the downloaded audio. (Allows YouTube links.)
    This version includes a progress hook to print download progress.
    """
    downloads_dir = os.path.join(os.getcwd(), "downloads")
    os.makedirs(downloads_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_template = os.path.join(downloads_dir, f"{timestamp}_%(title)s.%(ext)s")
    
    def progress_hook(d):
        if d.get("status") == "downloading":
            print(f"Downloading... {d.get('downloaded_bytes', 0)} bytes downloaded.", flush=True)
        elif d.get("status") == "finished":
            print("Download finished, now post-processing...", flush=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': desired_format,
            'preferredquality': '192',
        }],
        'quiet': False,
        'socket_timeout': 30,
        'progress_hooks': [progress_hook],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_filename = ydl.prepare_filename(info)
            base, ext = os.path.splitext(downloaded_filename)
            if ext.lower() != f".{desired_format}":
                downloaded_filename = base + f".{desired_format}"
            print(f"Audio downloaded to: {downloaded_filename}", flush=True)
            return downloaded_filename
    except Exception as e:
        print(f"Error in get_audio: {e}", flush=True)
        return f"Error downloading audio: {e}"

def prompt_for_audio_source() -> str:
    """
    Prompts the user whether to provide a local file or a link for audio.
    If a link is provided, it downloads the audio using get_audio.
    Returns the local file path to the audio.
    """
    mode = input("Use (1) file or (2) link? Enter 1 or 2: ").strip()
    if mode == "1":
        audio_path = input("Enter the path to your audio file: ").strip()
        return audio_path
    elif mode == "2":
        url = input("Enter the audio URL: ").strip()
        print("Choose audio format:")
        print("  1) MP3")
        print("  2) WAV")
        fmt_choice = input("Enter 1 or 2: ").strip()
        if fmt_choice == "1":
            desired_format = "mp3"
        elif fmt_choice == "2":
            desired_format = "wav"
        else:
            print("Invalid choice, defaulting to MP3.")
            desired_format = "mp3"
        downloaded = get_audio(url, desired_format=desired_format)
        if downloaded.startswith("Error downloading audio:"):
            print("This video cannot be downloaded.")
            return ""
        else:
            return downloaded
    else:
        print("Invalid choice.")
        return ""

def chat_with_api(prompt: str) -> str:
    """
    A simple wrapper to interact with GPT-4.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

###############################################
# MAIN APPLICATION LOOP
###############################################
def main():
    downloads_dir = os.path.join(os.getcwd(), "downloads")
    print("Welcome! I'm ThisIsMusic.ai, your digital music consultant.")
    print("I can assist with various music functions:")
    print("  /transcribe   - Transcribe audio from a file or link")
    print("  /lyrics       - Generate formatted lyrics (PDF) from audio (file or link)")
    print("  /article      - Generate an article (PDF) from audio (file or link)")
    print("  /summarize    - Summarize text from a file or URL (PDF)")
    print("  /getaudio     - Get audio from a URL (automatically downloaded)")
    print("  /pressrelease - Generate a press release (PDF) for a release")
    print("  /social       - Generate a social media post for a song")
    print("  /epk          - Generate an Electronic Press Kit (EPK) for an artist")
    print("  /chat         - Chat normally for directed help or consulting")
    print("Type 'exit' to quit the application.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Command: Transcribe audio
        if user_input.startswith("/transcribe"):
            audio_path = prompt_for_audio_source()
            if not audio_path:
                continue
            transcript = transcribe_audio(audio_path)
            print("\n--- Transcription ---")
            print(transcript)
            # Delete downloaded file if it was auto-downloaded
            if audio_path.startswith(downloads_dir):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    print(f"Error deleting file: {e}")

        # Command: Generate lyrics PDF from audio
        elif user_input.startswith("/lyrics"):
            audio_path = prompt_for_audio_source()
            if not audio_path:
                continue
            transcript = transcribe_audio(audio_path)
            output_text = generate_output(transcript, task="lyrics")
            pdf_filename = input("Enter output PDF filename for lyrics (e.g., lyrics.pdf): ").strip()
            text_to_pdf(output_text, pdf_filename)
            if audio_path.startswith(downloads_dir):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    print(f"Error deleting file: {e}")

        # Command: Generate article PDF from audio
        elif user_input.startswith("/article"):
            audio_path = prompt_for_audio_source()
            if not audio_path:
                continue
            transcript = transcribe_audio(audio_path)
            output_text = generate_output(transcript, task="article")
            pdf_filename = input("Enter output PDF filename for article (e.g., article.pdf): ").strip()
            text_to_pdf(output_text, pdf_filename)
            if audio_path.startswith(downloads_dir):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    print(f"Error deleting file: {e}")

        # Command: Summarize text from a file or URL and generate PDF
        elif user_input.startswith("/summarize"):
            choice = input("Summarize from (1) file or (2) URL? Enter 1 or 2: ").strip()
            if choice == "1":
                file_path = input("Enter the path to your text file: ").strip()
                try:
                    with open(file_path, "r") as f:
                        text = f.read()
                except Exception as e:
                    print(f"Error reading file: {e}")
                    continue
            elif choice == "2":
                url = input("Enter the URL: ").strip()
                try:
                    r = requests.get(url)
                    r.raise_for_status()
                    text = r.text
                except Exception as e:
                    print(f"Error downloading text: {e}")
                    continue
            else:
                print("Invalid choice.")
                continue

            output_text = generate_output(text, task="summarize")
            pdf_filename = input("Enter output PDF filename for summary (e.g., summary.pdf): ").strip()
            text_to_pdf(output_text, pdf_filename)

        # Command: Get Audio from URL
        elif user_input.startswith("/getaudio"):
            print("Choose audio format:")
            print("  1) MP3")
            print("  2) WAV")
            fmt_choice = input("Enter 1 or 2: ").strip()
            if fmt_choice == "1":
                desired_format = "mp3"
            elif fmt_choice == "2":
                desired_format = "wav"
            else:
                print("Invalid choice, defaulting to MP3.")
                desired_format = "mp3"
            source = input("Use (1) link or (2) file? Enter 1 or 2: ").strip()
            if source == "1":
                url = input("Enter the audio URL: ").strip()
                downloaded_path = get_audio(url, desired_format=desired_format)
                if downloaded_path.startswith("Error downloading audio:"):
                    print("This video cannot be downloaded.")
                    continue
            elif source == "2":
                downloaded_path = input("Enter the path to your audio file: ").strip()
                if not os.path.exists(downloaded_path):
                    print("File does not exist.")
                    continue
            else:
                print("Invalid choice.")
                continue

            if downloaded_path:
                print("Audio downloaded successfully!")
                process_choice = input("Would you like to process this audio? (1) Get Lyrics, (2) Make an Article, (3) No: ").strip()
                if process_choice == "1":
                    transcript = transcribe_audio(downloaded_path)
                    output_text = generate_output(transcript, task="lyrics")
                    pdf_filename = input("Enter output PDF filename for lyrics (e.g., lyrics.pdf): ").strip()
                    text_to_pdf(output_text, pdf_filename)
                    if downloaded_path.startswith(downloads_dir):
                        try:
                            os.remove(downloaded_path)
                        except Exception as e:
                            print(f"Error deleting file: {e}")
                elif process_choice == "2":
                    transcript = transcribe_audio(downloaded_path)
                    output_text = generate_output(transcript, task="article")
                    pdf_filename = input("Enter output PDF filename for article (e.g., article.pdf): ").strip()
                    text_to_pdf(output_text, pdf_filename)
                    if downloaded_path.startswith(downloads_dir):
                        try:
                            os.remove(downloaded_path)
                        except Exception as e:
                            print(f"Error deleting file: {e}")
                else:
                    print(f"Audio file available at: {downloaded_path}")

        # Command: Generate a Press Release (PDF)
        elif user_input.startswith("/pressrelease"):
            song_title = input("Enter the album or song title: ").strip()
            artist_name = input("Enter the artist name: ").strip()
            release_date = input("Enter the release date (YYYY-MM-DD): ").strip()
            album_description = input("Enter a brief album description: ").strip()
            output_text = generate_press_release(song_title, artist_name, release_date, album_description)
            pdf_filename = input("Enter output PDF filename for the press release (e.g., press.pdf): ").strip()
            text_to_pdf(output_text, pdf_filename)
        
        # Command: Generate a Social Media Post
        elif user_input.startswith("/social"):
            song_title = input("Enter the song title: ").strip()
            artist_name = input("Enter the artist name: ").strip()
            post = generate_social_media_post(song_title, artist_name)
            print("\n--- Social Media Post ---")
            print(post)
        
        # Command: Generate an EPK for the artist
        elif user_input.startswith("/epk"):
            artist_name = input("Enter the artist's name: ").strip()
            background_info = input("Enter background information about the artist: ").strip()
            achievements = input("Enter the artist's achievements: ").strip()
            social_links = input("Enter social media or website links (comma-separated): ").strip()
            press_quotes = input("Enter any press quotes (optional): ").strip()
            epk_text = generate_epk(artist_name, background_info, achievements, social_links, press_quotes)
            print("\n--- Electronic Press Kit (EPK) ---")
            print(epk_text)
            pdf_choice = input("Would you like to save this EPK as a PDF? (y/n): ").strip().lower()
            if pdf_choice == "y":
                pdf_filename = input("Enter output PDF filename for the EPK (e.g., epk.pdf): ").strip()
                text_to_pdf(epk_text, pdf_filename)
        
        # Command: Chat normally for consulting/directed help
        elif user_input.startswith("/chat"):
            prompt = input("Enter your message: ").strip()
            response = chat_with_api(prompt)
            print("ThisIsMusic.ai:", response, "\n")
        
        # Default: Continue conversation (direct consulting)
        else:
            response = chat_with_api(user_input)
            print("ThisIsMusic.ai:", response, "\n")

if __name__ == "__main__":
    main()
