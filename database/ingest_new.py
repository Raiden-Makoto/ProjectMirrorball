import os
import re
import lyricsgenius # type: ignore
import duckdb # type: ignore
import pandas as pd # type: ignore
from prefect import flow, task # type: ignore

from dotenv import load_dotenv # type: ignore
load_dotenv()

# --- CONFIGURATION ---
GENIUS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")
if not GENIUS_TOKEN:
    raise ValueError("GENIUS_ACCESS_TOKEN not found in environment. Make sure .env file exists and contains GENIUS_ACCESS_TOKEN")
DB_PATH = os.path.join(os.getcwd(), "mirrorball.db")
SEED_PATH = os.path.join(os.getcwd(), "data", "mirrorball_seed.csv")
# Albums not in your CSV that we need to "Discover"
NEW_ALBUMS = [
    "Speak Now (Taylor's Version)",
    "1989 (Taylor's Version)",
    "The Tortured Poets Department",
    "The Tortured Poets Department: The Anthology",
    "The Life of a Showgirl"
]

# --- DATABASE SETUP ---
def init_lyrics_table():
    conn = duckdb.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dim_lyrics (
            track_name TEXT,
            album_name TEXT,
            lyrics TEXT,
            is_new_era BOOLEAN,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.close()

# --- UTILITIES ---
def clean_lyrics(text):
    """Strips Genius tags like [Chorus], [Bridge], etc."""
    if not text: return None
    # Remove metadata lines like "176 Contributors", "Lyrics", etc.
    text = re.sub(r'.*?Lyrics', '', text, count=1)
    # Remove [Verse 1], [Chorus], etc.
    text = re.sub(r'\[.*?\]', '', text)
    # Remove the 'Embed' at the end of Genius lyrics
    text = re.sub(r'[0-9]*Embed$', '', text)
    return text.strip()

@task(retries=3, retry_delay_seconds=10)
def scrape_song_lyrics(genius, track_name, album_name, is_new):
    conn = duckdb.connect(DB_PATH)
    
    # CHECKPOINT: Skip if already scraped
    exists = conn.execute(
        "SELECT 1 FROM dim_lyrics WHERE track_name = ? AND album_name = ?", 
        [track_name, album_name]
    ).fetchone()
    
    if exists:
        conn.close()
        return f"Skipped: {track_name}"

    try:
        # Search for the song
        song = genius.search_song(track_name, "Taylor Swift")
        if song:
            lyrics = clean_lyrics(song.lyrics)
            conn.execute(
                "INSERT INTO dim_lyrics (track_name, album_name, lyrics, is_new_era) VALUES (?, ?, ?, ?)",
                [track_name, album_name, lyrics, is_new]
            )
            conn.close()
            return f"Scraped: {track_name}"
    except Exception as e:
        conn.close()
        return f"Error on {track_name}: {str(e)}"
    
    conn.close()
    return f"Not Found: {track_name}"

# --- THE MAIN FLOW ---
@flow(name="Mirrorball Ingestion")
def run_ingestion():
    init_lyrics_table()
    genius = lyricsgenius.Genius(GENIUS_TOKEN, sleep_time=2, timeout=15)
    
    # Part 1: Scrape tracks from your Seed CSV
    print("--- PHASE 1: SEED TRACKS ---")
    df_seed = pd.read_csv(SEED_PATH)
    for _, row in df_seed.iterrows():
        res = scrape_song_lyrics(genius, row['track_name'], row['album_name'], False)
        print(res)

    # Part 2: Discover and Scrape New Eras
    print("\n--- PHASE 2: NEW ERAS DISCOVERY ---")
    for album_name in NEW_ALBUMS:
        print(f"Searching Album: {album_name}...")
        try:
            # Genius can be picky; if it fails, try searching without "Taylor Swift" in the query
            album = genius.search_album(album_name, "Taylor Swift")
            
            if album:
                for track_item in album.tracks:
                    # FIX: Genius returns tracks as a tuple (track_number, song_object)
                    # We need the second element of the tuple
                    song = track_item[1] if isinstance(track_item, tuple) else track_item
                    
                    track_title = song.title
                    res = scrape_song_lyrics(genius, track_title, album_name, True)
                    print(res)
            else:
                print(f"Found no results for album: {album_name}")
                
        except Exception as e:
            print(f"Error processing album {album_name}: {e}")

if __name__ == "__main__":
    run_ingestion()