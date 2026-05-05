# 🎬 CineMatch — Movie Recommendation System

A content-based movie recommendation system built with Python and Streamlit.

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

---

## 📦 Dataset

The app **auto-downloads** the TMDB 5000 Movies dataset from GitHub mirrors on first run and saves it locally as CSV files. If auto-download fails:

1. Go to [Kaggle — TMDB Movie Metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
2. Download both files:
   - `tmdb_5000_movies.csv`
   - `tmdb_5000_credits.csv`
3. Place them in the **same folder as `app.py`**
4. Re-run the app — it will load from local files automatically

---

## 🧠 How It Works

| Step | Description |
|------|-------------|
| **Feature Engineering** | Combines plot overview, genres, keywords, top cast, and director into a single text "soup" |
| **TF-IDF Vectorization** | Converts text soup into a numeric matrix (10,000 features) |
| **Cosine Similarity** | Computes pairwise similarity scores between all movies |
| **Recommendation** | Returns top-N most similar movies (excluding the input) |

---

## ✨ Features
- 🔍 Search from 4800+ movies
- 🎯 Match % score for each recommendation
- 🖼️ Movie posters via TMDB CDN
- 📊 Browse & filter full catalog
- 📈 Data exploration charts
- ⚙️ Adjustable number of results & vote count filter

---

## 🗂️ Project Structure
```
movie_recommender/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🛠️ Tech Stack
- **Streamlit** — Web UI
- **Scikit-learn** — TF-IDF & Cosine Similarity
- **Pandas / NumPy** — Data processing
- **TMDB API** — Movie posters

---

## 💡 Possible Enhancements
- Collaborative filtering (user ratings)
- Hybrid recommendation (content + collaborative)
- User login & watchlist
- TMDB API integration for real-time data