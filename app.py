import os
import plotly.graph_objects as go # type: ignore
import plotly.express as px # type: ignore
import duckdb # type: ignore
import pandas as pd # type: ignore

# Get the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
DB_PATH = os.path.join(PROJECT_ROOT, "mirrorball.db")

def mirrorball_app():
    conn = duckdb.connect(DB_PATH) # type: ignore
    df = conn.execute("SELECT * FROM final_map_data_with_shap").df()
    
    # 1. Clean the labels for the legend
    df['album_name'] = df['album_name'].str.replace("(Taylor's Version)", "(TV)", regex=False)
    df['album_name'] = df['album_name'].str.replace("The Tortured Poets Department", "TTPD", regex=False)
    
    # Group both TTPD albums together - merge to "TTPD: The Anthology" (superset)
    df['album_name'] = df['album_name'].str.replace("TTPD", "TTPD: The Anthology", regex=False)
    
    # 2. Map Cluster IDs to Descriptions for the Hover
    archetype_map = {
        0: {"name": "Quill Pen", "desc": "Poetic & Archaic (High Complexity, Low Energy)"},
        1: {"name": "Fountain Pen", "desc": "Modern Confessional (Specific Storytelling)"},
        2: {"name": "Glitter Gel Pen", "desc": "Frivolous & Upbeat (High Energy, High Valence)"},
        3: {"name": "Revenge Anthem", "desc": "Angst & Power (High Energy, Low Valence)"},
        4: {"name": "Standard Pop", "desc": "Radio-Ready (Mid-Range Baseline Production)"}
    }
    
    df['archetype_name'] = df['cluster_id'].map(lambda x: archetype_map[x]['name'])
    df['archetype_desc'] = df['cluster_id'].map(lambda x: archetype_map[x]['desc'])

    # Map albums to valid Plotly symbols
    valid_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 
                     'triangle-down', 'star', 'hexagon', 'pentagon', 'octagon',
                     'hexagram', 'diamond-wide', 'hourglass', 'bowtie']
    unique_albums = sorted(df['album_name'].unique())
    album_to_symbol = {album: valid_symbols[i % len(valid_symbols)] 
                       for i, album in enumerate(unique_albums)}
    df['symbol'] = df['album_name'].map(album_to_symbol)

    fig = go.Figure()

    # 3. Add the Main Data Trace
    fig.add_trace(go.Scatter(
        x=df['umap_x'],
        y=df['umap_y'],
        mode='markers',
        marker=dict(
            size=11,
            color=df['cluster_id'],
            colorscale='Viridis',
            symbol=df['symbol'],
            line=dict(width=0.5, color='white'),
            opacity=0.8
        ),
        text=df['track_name'],
        customdata=df[['album_name', 'archetype_name', 'archetype_desc', 'top_driver']],
        hovertemplate=(
            "<b>%{text}</b><br>" +
            "<i>%{customdata[0]}</i><br><br>" +
            "<b>Archetype:</b> %{customdata[1]}<br>" +
            "<b>Vibe:</b> %{customdata[2]}<br>" +
            "<b>Top Driver:</b> %{customdata[3]}<extra></extra>"
        ),
        showlegend=False
    ))

    # 4. SECOND LEGEND: The Archetypes (Colors)
    for cid, info in archetype_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            name=f"<b>{info['name']}</b>",
            marker=dict(size=12, color=px.colors.sequential.Viridis[cid*2]),
            legendgroup="Archetypes",
            legendgrouptitle_text="ML SONGWRITING ARCHETYPES"
        ))

    # 5. FIRST LEGEND: The Eras (Shapes)
    for album in unique_albums:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            name=album,
            marker=dict(size=10, color='white', symbol=album_to_symbol[album]),
            legendgroup="Eras",
            legendgrouptitle_text="ALBUM ERAS (SHAPES)"
        ))

    # 6. Layout Perfection
    fig.update_layout(
        template="plotly_dark",
        title="<b>Project Mirrorball: The Taylor Swift Latent Space Browser</b>",
        legend=dict(traceorder="grouped", x=1.02, y=1, font=dict(size=10)),
        margin=dict(l=20, r=220, t=80, b=20),
        width=1300, height=850,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
        paper_bgcolor='rgb(17,17,17)'   # Dark background covering entire page
    )

    fig.show()
    fig.write_html("mirrorball_map.html")
    print("Standalone map saved as mirrorball_map.html")
    
    # Export final table as CSV
    df.to_csv("mirrorball.csv", index=False)
    print("Final table exported as mirrorball.csv")
    
    conn.close()

if __name__ == "__main__":
    mirrorball_app()