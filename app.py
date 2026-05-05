import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import ast
import os
import io

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root tokens ── */
:root {
  --bg: #0d0d0d;
  --surface: #161616;
  --card: #1e1e1e;
  --border: #2a2a2a;
  --accent: #e8b84b;
  --accent2: #e05252;
  --text: #f0ede8;
  --muted: #888;
  --radius: 12px;
}

/* ── Global reset ── */
html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1300px; }

/* ── Hero title ── */
.hero-title {
  font-family: 'Bebas Neue', sans-serif;
  font-size: clamp(3rem, 7vw, 6rem);
  letter-spacing: 4px;
  line-height: 1;
  color: var(--text);
  margin: 0;
}
.hero-title span { color: var(--accent); }
.hero-sub {
  font-size: 1rem;
  color: var(--muted);
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-top: 0.4rem;
}
.hero-divider {
  width: 80px; height: 3px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  border-radius: 2px;
  margin: 1rem 0 2rem;
}

/* ── Metric badges ── */
.metrics-row { display: flex; gap: 1rem; margin: 1.5rem 0; flex-wrap: wrap; }
.metric-badge {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.8rem 1.4rem;
  text-align: center;
  min-width: 120px;
}
.metric-badge .val {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 2rem;
  color: var(--accent);
  line-height: 1;
}
.metric-badge .lbl { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }

/* ── Movie card grid ── */
.movie-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1.2rem; margin-top: 1.5rem; }
.movie-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  transition: transform 0.2s ease, border-color 0.2s ease;
  position: relative;
}
.movie-card:hover { transform: translateY(-4px); border-color: var(--accent); }
.movie-card img { width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block; }
.movie-card .no-poster {
  width: 100%; aspect-ratio: 2/3;
  background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
  display: flex; align-items: center; justify-content: center;
  font-size: 3rem;
}
.movie-card .card-body { padding: 0.8rem; }
.movie-card .card-title { font-size: 0.88rem; font-weight: 600; line-height: 1.3; margin-bottom: 0.3rem; }
.movie-card .card-meta { font-size: 0.75rem; color: var(--muted); }
.card-rank {
  position: absolute; top: 8px; left: 8px;
  background: var(--accent); color: #000;
  font-family: 'Bebas Neue', sans-serif;
  font-size: 0.9rem;
  padding: 2px 8px; border-radius: 6px;
}
.score-bar-wrap { margin-top: 0.5rem; }
.score-label { font-size: 0.7rem; color: var(--muted); display: flex; justify-content: space-between; }
.score-bar { height: 3px; background: var(--border); border-radius: 2px; margin-top: 3px; }
.score-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, var(--accent), var(--accent2)); }

/* ── Selectbox & input overrides ── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] input {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: var(--radius) !important;
}
div[data-baseweb="popover"] { background: var(--card) !important; }
li[role="option"] { background: var(--card) !important; color: var(--text) !important; }
li[role="option"]:hover { background: var(--border) !important; }

/* ── Slider ── */
div[data-testid="stSlider"] > div > div > div { background: var(--accent) !important; }

/* ── Section headings ── */
.section-head {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 1.6rem;
  letter-spacing: 3px;
  color: var(--text);
  border-left: 4px solid var(--accent);
  padding-left: 12px;
  margin: 2rem 0 1rem;
}

/* ── Tag pills ── */
.genre-pill {
  display: inline-block;
  background: rgba(232,184,75,0.12);
  border: 1px solid rgba(232,184,75,0.3);
  color: var(--accent);
  font-size: 0.7rem;
  padding: 2px 8px;
  border-radius: 20px;
  margin: 2px 2px 0 0;
  font-weight: 500;
  letter-spacing: 0.5px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] label { color: var(--muted) !important; font-size: 0.82rem !important; text-transform: uppercase; letter-spacing: 1px; }

/* ── Buttons ── */
.stButton > button {
  background: var(--accent) !important;
  color: #000 !important;
  font-family: 'Bebas Neue', sans-serif !important;
  font-size: 1.05rem !important;
  letter-spacing: 2px !important;
  border: none !important;
  border-radius: var(--radius) !important;
  padding: 0.6rem 2rem !important;
  transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Alert / info boxes ── */
.stAlert { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; color: var(--text) !important; }

/* ── Expander ── */
details { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }
summary { color: var(--text) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Tabs ── */
button[data-baseweb="tab"] { color: var(--muted) !important; background: transparent !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DATA LOADING  (TMDB 5000 via GitHub)
# ─────────────────────────────────────────────
POSTER_BASE = "https://image.tmdb.org/t/p/w500"

# Multiple mirror URLs to try in order
MOVIES_URLS = [
    "https://raw.githubusercontent.com/YBIFoundation/Dataset/main/tmdb_5000_movies.csv",
    "https://raw.githubusercontent.com/rashida048/Datasets/master/tmdb_5000_movies.csv",
    "https://raw.githubusercontent.com/erkansirin78/datasets/master/tmdb_5000_movies.csv",
]
CREDITS_URLS = [
    "https://raw.githubusercontent.com/YBIFoundation/Dataset/main/tmdb_5000_credits.csv",
    "https://raw.githubusercontent.com/rashida048/Datasets/master/tmdb_5000_credits.csv",
    "https://raw.githubusercontent.com/erkansirin78/datasets/master/tmdb_5000_credits.csv",
]

LOCAL_MOVIES = "tmdb_5000_movies.csv"
LOCAL_CREDITS = "tmdb_5000_credits.csv"


def safe_literal(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return []


def try_download_csv(urls, label):
    """Try multiple URLs; return DataFrame or None."""
    for url in urls:
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                return pd.read_csv(io.StringIO(resp.text))
        except Exception:
            continue
    return None


@st.cache_data(show_spinner=False)
def load_data():
    try:
        # ── Try local files first (after first successful download) ──
        if os.path.exists(LOCAL_MOVIES) and os.path.exists(LOCAL_CREDITS):
            movies = pd.read_csv(LOCAL_MOVIES)
            credits = pd.read_csv(LOCAL_CREDITS)
        else:
            # ── Download from mirrors ──
            movies = try_download_csv(MOVIES_URLS, "movies")
            credits = try_download_csv(CREDITS_URLS, "credits")

            if movies is None or credits is None:
                st.error(
                    "❌ Could not download the dataset from any mirror.\n\n"
                    "**Manual fix (takes 1 minute):**\n"
                    "1. Download both CSV files from [this Kaggle page](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)\n"
                    "2. Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the **same folder as `app.py`**\n"
                    "3. Re-run the app"
                )
                return pd.DataFrame()

            # Cache locally for next run
            try:
                movies.to_csv(LOCAL_MOVIES, index=False)
                credits.to_csv(LOCAL_CREDITS, index=False)
            except Exception:
                pass

        # Rename for merge
        if "movie_id" in credits.columns:
            credits = credits.rename(columns={"movie_id": "id"})
        elif "id" not in credits.columns and "title" in credits.columns:
            credits = credits.rename(columns={"title": "credit_title"})

        # Merge
        if "id" in credits.columns:
            df = movies.merge(credits, on="id", how="left")
        else:
            df = movies.copy()

        # Clean title column
        if "title_x" in df.columns:
            df["title"] = df["title_x"]
        elif "title" not in df.columns:
            df["title"] = df.get("original_title", "Unknown")

        # Parse list columns
        for col in ["genres", "keywords"]:
            if col in df.columns:
                df[col] = df[col].fillna("[]").apply(safe_literal)
                df[col + "_str"] = df[col].apply(
                    lambda x: " ".join([i["name"].replace(" ", "") for i in x if isinstance(i, dict)])
                )
            else:
                df[col + "_str"] = ""

        for col in ["cast", "crew"]:
            if col in df.columns:
                df[col] = df[col].fillna("[]").apply(safe_literal)
            else:
                df[col] = [[]]

        # Top 3 cast
        df["cast_str"] = df["cast"].apply(
            lambda x: " ".join([i["name"].replace(" ", "") for i in x[:3] if isinstance(i, dict)])
        )

        # Director
        def get_director(crew):
            for c in crew:
                if isinstance(c, dict) and c.get("job") == "Director":
                    return c["name"].replace(" ", "")
            return ""

        df["director"] = df["crew"].apply(get_director)

        # Overview
        df["overview"] = df.get("overview", pd.Series([""] * len(df))).fillna("")

        # Soup feature
        df["soup"] = (
            df["overview"].str.lower().str.replace(r"[^a-z\s]", "", regex=True)
            + " "
            + df["genres_str"]
            + " "
            + df["keywords_str"]
            + " "
            + df["cast_str"]
            + " "
            + df["director"]
        )

        # Keep useful columns
        keep = ["id", "title", "soup", "genres_str", "vote_average",
                "vote_count", "release_date", "overview",
                "poster_path" if "poster_path" in df.columns else "id"]
        keep = [c for c in keep if c in df.columns]
        df = df[keep].dropna(subset=["title"]).drop_duplicates(subset=["title"]).reset_index(drop=True)

        # Ensure poster_path
        if "poster_path" not in df.columns:
            df["poster_path"] = None

        # Year
        if "release_date" in df.columns:
            df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
        else:
            df["year"] = None

        return df
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def build_similarity(df):
    tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
    matrix = tfidf.fit_transform(df["soup"].fillna(""))
    sim = cosine_similarity(matrix, matrix)
    return sim


def get_recommendations(title, df, sim_matrix, n=10, min_votes=50):
    indices = pd.Series(df.index, index=df["title"].str.lower())
    key = title.strip().lower()
    if key not in indices:
        return pd.DataFrame()
    idx = indices[key]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:]  # exclude self

    results = []
    for i, score in scores:
        row = df.iloc[i]
        if row.get("vote_count", 0) >= min_votes or pd.isna(row.get("vote_count")):
            results.append({
                "title": row["title"],
                "score": round(score * 100, 1),
                "genres": row.get("genres_str", ""),
                "vote_avg": row.get("vote_average", "N/A"),
                "year": row.get("year", ""),
                "overview": row.get("overview", ""),
                "poster_path": row.get("poster_path"),
            })
        if len(results) == n:
            break

    return pd.DataFrame(results)


def poster_url(path):
    if path and isinstance(path, str) and path.startswith("/"):
        return POSTER_BASE + path
    return None


def genre_pills(genre_str):
    if not genre_str:
        return ""
    genres = genre_str.strip().split()[:4]
    pills = "".join([f'<span class="genre-pill">{g}</span>' for g in genres])
    return pills


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    n_recs = st.slider("Number of recommendations", 5, 20, 10)
    min_votes = st.slider("Min. vote count filter", 0, 500, 50, step=10)

    st.markdown("---")
    st.markdown("### 📊 Algorithm")
    st.markdown("""
**Content-Based Filtering** using TF-IDF vectorization on:
- Movie overview/plot
- Genres & keywords
- Top cast members
- Director

Similarity measured via **Cosine Similarity**.
    """)
    st.markdown("---")
    st.markdown("### 💡 How to Use")
    st.markdown("""
1. Search or select a movie you like
2. Adjust filters in settings
3. Hit **Find Similar Movies**
4. Explore your recommendations!
    """)


# ─────────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-title">CINE<span>MATCH</span></div>
<div class="hero-sub">Content-Based Movie Recommendation Engine</div>
<div class="hero-divider"></div>
""", unsafe_allow_html=True)

# Load data
with st.spinner("Loading movie dataset..."):
    df = load_data()

if df.empty:
    st.error("Could not load movie data. Please check your internet connection.")
    st.stop()

with st.spinner("Building similarity matrix..."):
    sim_matrix = build_similarity(df)

# Metrics
total_movies = len(df)
avg_rating = df["vote_average"].mean() if "vote_average" in df.columns else 0
genres_count = df["genres_str"].str.split().explode().nunique() if "genres_str" in df.columns else 0

st.markdown(f"""
<div class="metrics-row">
  <div class="metric-badge"><div class="val">{total_movies:,}</div><div class="lbl">Movies</div></div>
  <div class="metric-badge"><div class="val">{avg_rating:.1f}</div><div class="lbl">Avg Rating</div></div>
  <div class="metric-badge"><div class="val">{genres_count}</div><div class="lbl">Genres</div></div>
  <div class="metric-badge"><div class="val">TF‑IDF</div><div class="lbl">Algorithm</div></div>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ───
tab1, tab2, tab3 = st.tabs(["🔍  Recommend", "📋  Browse Movies", "📈  Explore Data"])

# ══════════════════════════════════════════════
#  TAB 1 — RECOMMEND
# ══════════════════════════════════════════════
with tab1:
    col_search, col_btn = st.columns([4, 1])
    with col_search:
        movie_titles = sorted(df["title"].dropna().tolist())
        selected_movie = st.selectbox(
            "Search for a movie you love:",
            options=[""] + movie_titles,
            index=0,
            format_func=lambda x: "— Type to search —" if x == "" else x,
        )
    with col_btn:
        st.markdown("<div style='margin-top:1.8rem'></div>", unsafe_allow_html=True)
        search_btn = st.button("🎯 Find Similar")

    # Show selected movie details
    if selected_movie:
        movie_row = df[df["title"] == selected_movie].iloc[0]
        with st.expander(f"📽️  About: {selected_movie}", expanded=True):
            c1, c2 = st.columns([1, 3])
            with c1:
                url = poster_url(movie_row.get("poster_path"))
                if url:
                    st.image(url, use_column_width=True)
                else:
                    st.markdown("<div style='font-size:4rem;text-align:center'>🎬</div>", unsafe_allow_html=True)
            with c2:
                yr = int(movie_row["year"]) if pd.notna(movie_row.get("year")) else "N/A"
                rating = movie_row.get("vote_average", "N/A")
                st.markdown(f"**{selected_movie}** ({yr})")
                st.markdown(f"⭐ {rating}/10")
                st.markdown(genre_pills(movie_row.get("genres_str", "")), unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:0.9rem;color:#aaa;margin-top:0.7rem'>{movie_row.get('overview','')}</p>", unsafe_allow_html=True)

    # Recommendations
    if search_btn and selected_movie:
        with st.spinner("Finding your perfect movies..."):
            recs = get_recommendations(selected_movie, df, sim_matrix, n=n_recs, min_votes=min_votes)

        if recs.empty:
            st.warning("No recommendations found. Try a different movie or lower the min votes filter.")
        else:
            st.markdown('<div class="section-head">RECOMMENDED FOR YOU</div>', unsafe_allow_html=True)

            # Build card HTML
            cards_html = '<div class="movie-grid">'
            for rank, (_, row) in enumerate(recs.iterrows(), 1):
                url = poster_url(row["poster_path"])
                img_tag = (
                    f'<img src="{url}" alt="{row["title"]}" loading="lazy">'
                    if url
                    else '<div class="no-poster">🎬</div>'
                )
                yr = int(row["year"]) if pd.notna(row.get("year")) else ""
                score_pct = min(row["score"], 100)
                genre_html = genre_pills(row.get("genres", ""))
                cards_html += f"""
                <div class="movie-card">
                  <div class="card-rank">#{rank}</div>
                  {img_tag}
                  <div class="card-body">
                    <div class="card-title">{row['title']}</div>
                    <div class="card-meta">{'⭐ ' + str(row['vote_avg']) if row['vote_avg'] != 'N/A' else ''} {'· ' + str(yr) if yr else ''}</div>
                    {genre_html}
                    <div class="score-bar-wrap">
                      <div class="score-label"><span>Match</span><span>{score_pct}%</span></div>
                      <div class="score-bar"><div class="score-fill" style="width:{score_pct}%"></div></div>
                    </div>
                  </div>
                </div>"""
            cards_html += "</div>"
            st.markdown(cards_html, unsafe_allow_html=True)

            # Table view toggle
            with st.expander("📊 View as table"):
                display_df = recs[["title", "score", "vote_avg", "year", "genres"]].copy()
                display_df.columns = ["Title", "Match %", "Rating", "Year", "Genres"]
                st.dataframe(display_df, use_container_width=True, hide_index=True)

    elif search_btn and not selected_movie:
        st.warning("Please select a movie first!")

# ══════════════════════════════════════════════
#  TAB 2 — BROWSE
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-head">BROWSE MOVIES</div>', unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    with b1:
        search_text = st.text_input("🔎 Search title", "")
    with b2:
        min_rating = st.slider("Min rating", 0.0, 10.0, 6.0, 0.5)
    with b3:
        sort_by = st.selectbox("Sort by", ["vote_average", "vote_count", "title"])

    filtered = df.copy()
    if search_text:
        filtered = filtered[filtered["title"].str.lower().str.contains(search_text.lower(), na=False)]
    if "vote_average" in filtered.columns:
        filtered = filtered[filtered["vote_average"] >= min_rating]
    if sort_by in filtered.columns:
        asc = sort_by == "title"
        filtered = filtered.sort_values(sort_by, ascending=asc)

    st.markdown(f"<p style='color:var(--muted);font-size:0.85rem'>Showing {min(len(filtered),50)} of {len(filtered)} movies</p>", unsafe_allow_html=True)

    show_cols = [c for c in ["title", "vote_average", "year", "genres_str"] if c in filtered.columns]
    st.dataframe(
        filtered[show_cols].head(50).rename(columns={"vote_average": "Rating", "genres_str": "Genres", "title": "Title", "year": "Year"}),
        use_container_width=True,
        hide_index=True,
    )

# ══════════════════════════════════════════════
#  TAB 3 — EXPLORE DATA
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-head">DATA EXPLORATION</div>', unsafe_allow_html=True)

    e1, e2 = st.columns(2)

    with e1:
        st.markdown("**Rating Distribution**")
        if "vote_average" in df.columns:
            hist_data = df["vote_average"].dropna()
            st.bar_chart(hist_data.value_counts().sort_index(), color="#e8b84b")

    with e2:
        st.markdown("**Top Genres**")
        if "genres_str" in df.columns:
            genre_series = df["genres_str"].str.split().explode().value_counts().head(15)
            st.bar_chart(genre_series, color="#e05252")

    e3, e4 = st.columns(2)
    with e3:
        st.markdown("**Movies by Decade**")
        if "year" in df.columns:
            decade_df = df["year"].dropna().astype(int)
            decade_df = ((decade_df // 10) * 10).value_counts().sort_index()
            decade_df.index = decade_df.index.astype(str) + "s"
            st.bar_chart(decade_df, color="#7ecba1")

    with e4:
        st.markdown("**Dataset Stats**")
        stats = {
            "Total Movies": len(df),
            "Average Rating": f"{df['vote_average'].mean():.2f}" if "vote_average" in df.columns else "N/A",
            "Highest Rated": df.loc[df["vote_average"].idxmax(), "title"] if "vote_average" in df.columns else "N/A",
            "Most Votes": df.loc[df["vote_count"].idxmax(), "title"] if "vote_count" in df.columns else "N/A",
        }
        for k, v in stats.items():
            st.metric(k, v)