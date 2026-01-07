"""
Lyrics Ingestion Pipeline for Project Mirrorball.

This module scrapes song lyrics from Genius API and stores them in DuckDB.
It handles two phases:
1. Scraping tracks from the seed CSV (legacy tracks)
2. Discovering and scraping new era albums (Taylor's Version, TTPD, etc.)

Uses Prefect for workflow orchestration with retry logic and error handling.
"""

import os
import re
from typing import Optional

import duckdb  # type: ignore
import lyricsgenius  # type: ignore
import pandas as pd  # type: ignore
from dotenv import load_dotenv  # type: ignore
from prefect import flow, task  # type: ignore

# Load environment variables
load_dotenv()

# Get the project root directory (one level up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "mirrorball.db")
SEED_PATH = os.path.join(PROJECT_ROOT, "data", "mirrorball_seed.csv")

# --- CONFIGURATION ---
GENIUS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")
if not GENIUS_TOKEN:
    raise ValueError(
        "GENIUS_ACCESS_TOKEN not found in environment. "
        "Make sure .env file exists and contains GENIUS_ACCESS_TOKEN"
    )

# Albums not in the seed CSV that we need to "Discover"
NEW_ALBUMS = [
    "Speak Now (Taylor's Version)",
    "1989 (Taylor's Version)",
    "The Tortured Poets Department",
    "The Tortured Poets Department: The Anthology",
    "The Life of a Showgirl",
]


# --- DATABASE SETUP ---
def init_lyrics_table() -> None:
    """
    Initialize the lyrics table in DuckDB if it doesn't exist.

    Creates dim_lyrics table with columns:
    - track_name: Song title
    - album_name: Album name
    - lyrics: Full lyrics text
    - is_new_era: Boolean flag for new era tracks
    - scraped_at: Timestamp of when lyrics were scraped
    """
    conn = duckdb.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dim_lyrics (
            track_name TEXT,
            album_name TEXT,
            lyrics TEXT,
            is_new_era BOOLEAN,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.close()


# --- UTILITIES ---
def clean_lyrics(text: Optional[str]) -> Optional[str]:
    """
    Clean lyrics text by removing Genius-specific metadata while preserving structure.

    Removes:
    - Metadata lines like "176 Contributors", "Lyrics"
    - Embed suffixes like "21Embed" at the end

    Preserves:
    - Section tags like [Verse 1], [Chorus], [Bridge] (needed for bridge analysis)

    Args:
        text: Raw lyrics text from Genius API

    Returns:
        Cleaned lyrics text with section tags preserved, or None if input is None/empty
    """
    if not text:
        return None

    # Remove metadata lines like "176 Contributors", "Lyrics", etc.
    text = re.sub(r".*?Lyrics", "", text, count=1)

    # NOTE: We intentionally KEEP [Verse], [Chorus], [Bridge] tags
    # because they are needed for bridge impact analysis in bridge_impact.py

    # Remove the 'Embed' at the end of Genius lyrics (e.g., "21Embed")
    text = re.sub(r"[0-9]*Embed$", "", text)

    return text.strip()


@task(retries=3, retry_delay_seconds=10)
def scrape_song_lyrics(
    genius: lyricsgenius.Genius,
    track_name: str,
    album_name: str,
    is_new: bool,
) -> str:
    """
    Scrape lyrics for a single song from Genius API.

    This function:
    1. Checks if lyrics already exist in database (skip if found)
    2. Searches for the song on Genius
    3. Cleans the lyrics text
    4. Saves to database

    Args:
        genius: Initialized Genius API client
        track_name: Name of the track to scrape
        album_name: Album name for the track
        is_new: Whether this is a new era track (True) or legacy (False)

    Returns:
        Status message: "Skipped: {track_name}", "Scraped: {track_name}",
        "Not Found: {track_name}", or "Error on {track_name}: {error}"
    """
    conn = duckdb.connect(DB_PATH)

    # CHECKPOINT: Skip if already scraped AND has section tags
    existing = conn.execute(
        "SELECT lyrics FROM dim_lyrics WHERE track_name = ? AND album_name = ?",
        [track_name, album_name],
    ).fetchone()

    if existing:
        lyrics = existing[0]
        # Check if lyrics have section tags (needed for bridge analysis)
        has_tags = (
            lyrics
            and (
                "[Verse" in lyrics
                or "[Chorus" in lyrics
                or "[Bridge" in lyrics
                or "[Intro" in lyrics
                or "[Outro" in lyrics
            )
        )
        if has_tags:
            conn.close()
            return f"Skipped: {track_name} (has tags)"
        else:
            # Re-scrape if missing tags
            conn.execute(
                "DELETE FROM dim_lyrics WHERE track_name = ? AND album_name = ?",
                [track_name, album_name],
            )
            conn.close()
            # Continue to scrape below

    try:
        # Search for the song
        song = genius.search_song(track_name, "Taylor Swift")
        if song:
            lyrics = clean_lyrics(song.lyrics)
            conn.execute(
                "INSERT INTO dim_lyrics (track_name, album_name, lyrics, is_new_era) VALUES (?, ?, ?, ?)",
                [track_name, album_name, lyrics, is_new],
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
def run_ingestion() -> None:
    """
    Main ingestion workflow for scraping lyrics.

    This flow:
    1. Initializes the database table
    2. Scrapes all tracks from the seed CSV (Phase 1)
    3. Discovers and scrapes new era albums (Phase 2)

    Uses Prefect for orchestration with automatic retries on failures.
    """
    init_lyrics_table()
    genius = lyricsgenius.Genius(GENIUS_TOKEN, sleep_time=2, timeout=15)

    # Part 1: Scrape tracks from your Seed CSV
    print("--- PHASE 1: SEED TRACKS ---")
    df_seed = pd.read_csv(SEED_PATH)
    for _, row in df_seed.iterrows():
        res = scrape_song_lyrics(
            genius, row["track_name"], row["album_name"], False
        )
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
                    song = (
                        track_item[1]
                        if isinstance(track_item, tuple)
                        else track_item
                    )

                    track_title = song.title
                    res = scrape_song_lyrics(genius, track_title, album_name, True)
                    print(res)
            else:
                print(f"Found no results for album: {album_name}")

        except Exception as e:
            print(f"Error processing album {album_name}: {e}")


if __name__ == "__main__":
    run_ingestion()
