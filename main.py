import ast
import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# =========================
# ENV
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

# Load the backend env file relative to this module so startup works from any cwd.
load_dotenv(ENV_PATH)
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError(
        f"TMDB_API_KEY missing. Put it in {ENV_PATH} as TMDB_API_KEY=xxxx"
    )


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Movie Recommender API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# PICKLE GLOBALS
# =========================
DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "movies_metadata.csv")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None
metadata_df: Optional[pd.DataFrame] = None
metadata_by_id: Optional[pd.DataFrame] = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None


# =========================
# MODELS
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = Field(default_factory=list)


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


# =========================
# UTILS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


def optional_str(value: Any) -> Optional[str]:
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    if value is None:
        return None

    text = str(value).strip()
    return text or None


def optional_float(value: Any) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_genres(value: Any) -> List[dict]:
    if isinstance(value, list):
        items = value
    else:
        text = optional_str(value)
        if not text or text == "[]":
            return []
        try:
            items = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return []

    if not isinstance(items, list):
        return []

    parsed: List[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        name = optional_str(item.get("name"))
        raw_id = item.get("id")
        if not name or raw_id is None:
            continue

        try:
            genre_id = int(raw_id)
        except (TypeError, ValueError):
            continue

        parsed.append({"id": genre_id, "name": name})

    return parsed


def extract_genre_ids(genres: List[dict]) -> Tuple[int, ...]:
    out: List[int] = []
    for genre in genres:
        raw_id = genre.get("id")
        if raw_id is None:
            continue
        try:
            out.append(int(raw_id))
        except (TypeError, ValueError):
            continue
    return tuple(out)


def build_local_metadata_catalog() -> pd.DataFrame:
    if not os.path.exists(METADATA_PATH):
        return pd.DataFrame()

    meta = pd.read_csv(
        METADATA_PATH,
        low_memory=False,
        usecols=[
            "id",
            "title",
            "overview",
            "genres",
            "poster_path",
            "release_date",
            "vote_average",
            "vote_count",
            "popularity",
        ],
    )

    meta["tmdb_id"] = pd.to_numeric(meta["id"], errors="coerce")
    meta["title"] = meta["title"].fillna("").astype(str).str.strip()
    meta["release_date"] = meta["release_date"].fillna("").astype(str).str.strip()
    meta["vote_average"] = pd.to_numeric(meta["vote_average"], errors="coerce")
    meta["vote_count"] = pd.to_numeric(meta["vote_count"], errors="coerce").fillna(0.0)
    meta["popularity"] = pd.to_numeric(meta["popularity"], errors="coerce").fillna(0.0)
    meta["genres_parsed"] = meta["genres"].apply(parse_genres)
    meta["genre_ids"] = meta["genres_parsed"].apply(extract_genre_ids)

    meta = meta.dropna(subset=["tmdb_id"]).copy()
    meta["tmdb_id"] = meta["tmdb_id"].astype(int)
    meta = meta[meta["title"] != ""].copy()

    return (
        meta.sort_values(
            by=["popularity", "vote_average", "vote_count", "title"],
            ascending=[False, False, False, True],
        )
        .drop_duplicates(subset=["tmdb_id"], keep="first")
        .reset_index(drop=True)
    )


def tmdb_error_fallback_ok(exc: HTTPException) -> bool:
    return exc.status_code == 502


def local_row_to_card(row: pd.Series) -> TMDBMovieCard:
    return TMDBMovieCard(
        tmdb_id=int(row["tmdb_id"]),
        title=optional_str(row.get("title")) or "Untitled",
        poster_url=make_img_url(optional_str(row.get("poster_path"))),
        release_date=optional_str(row.get("release_date")),
        vote_average=optional_float(row.get("vote_average")),
    )


def local_row_to_details(row: pd.Series) -> TMDBMovieDetails:
    return TMDBMovieDetails(
        tmdb_id=int(row["tmdb_id"]),
        title=optional_str(row.get("title")) or "Untitled",
        overview=optional_str(row.get("overview")),
        release_date=optional_str(row.get("release_date")),
        poster_url=make_img_url(optional_str(row.get("poster_path"))),
        backdrop_url=None,
        genres=row.get("genres_parsed") or [],
    )


def get_local_movie_row(tmdb_id: int) -> Optional[pd.Series]:
    global metadata_by_id

    if metadata_by_id is None or metadata_by_id.empty:
        return None

    try:
        row = metadata_by_id.loc[int(tmdb_id)]
    except KeyError:
        return None

    if isinstance(row, pd.DataFrame):
        return row.iloc[0]
    return row


def local_find_movies(query: str) -> pd.DataFrame:
    global metadata_df

    if metadata_df is None or metadata_df.empty:
        return pd.DataFrame()

    keyword = _norm_title(query)
    if not keyword:
        return pd.DataFrame()

    titles = metadata_df["title"].str.lower()

    exact = metadata_df[titles == keyword]
    if not exact.empty:
        return exact

    prefix = metadata_df[titles.str.startswith(keyword, na=False)]
    if not prefix.empty:
        return prefix

    return metadata_df[titles.str.contains(keyword, na=False)]


def local_find_best_movie(query: str) -> Optional[pd.Series]:
    matches = local_find_movies(query)
    if matches.empty:
        return None
    return matches.iloc[0]


def local_search_movies(query: str, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
    matches = local_find_movies(query)
    total_results = int(len(matches))
    total_pages = (total_results + page_size - 1) // page_size if total_results else 0

    if total_results == 0:
        return {
            "page": page,
            "results": [],
            "total_pages": 0,
            "total_results": 0,
        }

    start = (page - 1) * page_size
    page_rows = matches.iloc[start : start + page_size]
    results = []

    for _, row in page_rows.iterrows():
        results.append(
            {
                "id": int(row["tmdb_id"]),
                "title": optional_str(row.get("title")) or "Untitled",
                "poster_path": optional_str(row.get("poster_path")),
                "release_date": optional_str(row.get("release_date")) or "",
                "vote_average": optional_float(row.get("vote_average")),
            }
        )

    return {
        "page": page,
        "results": results,
        "total_pages": total_pages,
        "total_results": total_results,
    }


def local_home_cards(category: str, limit: int) -> List[TMDBMovieCard]:
    global metadata_df

    if metadata_df is None or metadata_df.empty:
        return []

    if category == "top_rated":
        rows = metadata_df.sort_values(
            by=["vote_average", "vote_count", "popularity", "title"],
            ascending=[False, False, False, True],
        )
    elif category in {"upcoming", "now_playing"}:
        rows = metadata_df.sort_values(
            by=["release_date", "popularity", "vote_average", "title"],
            ascending=[False, False, False, True],
        )
    else:
        rows = metadata_df

    cards: List[TMDBMovieCard] = []
    for _, row in rows.head(limit).iterrows():
        cards.append(local_row_to_card(row))
    return cards


def local_genre_recommendations(
    source_row: Optional[pd.Series],
    limit: int,
    exclude_tmdb_id: Optional[int] = None,
) -> List[TMDBMovieCard]:
    global metadata_df

    if source_row is None or metadata_df is None or metadata_df.empty:
        return []

    genre_ids = source_row.get("genre_ids") or ()
    if not genre_ids:
        return []

    first_genre_id = int(genre_ids[0])
    matches = metadata_df[
        metadata_df["genre_ids"].apply(lambda ids: first_genre_id in ids)
    ]

    if exclude_tmdb_id is not None:
        matches = matches[matches["tmdb_id"] != int(exclude_tmdb_id)]

    cards: List[TMDBMovieCard] = []
    for _, row in matches.head(limit).iterrows():
        cards.append(local_row_to_card(row))
    return cards


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safe TMDB GET:
    - Network errors -> 502
    - TMDB API errors -> 502 with detail
    """
    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB request error for {path}: {type(e).__name__} | {repr(e)}",
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"TMDB error {r.status_code}: {r.text}"
        )

    return r.json()


async def tmdb_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    """
    Raw TMDB response for keyword search (MULTIPLE results).
    Streamlit will use this for suggestions and grid.
    """
    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


# =========================
# TF-IDF Helpers
# =========================
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    indices.pkl can be:
    - dict(title -> index)
    - pandas Series (index=title, value=index)
    We normalize into TITLE_TO_IDX.
    """
    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx

    # pandas Series or similar mapping
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        # last resort: if it's a list-like etc.
        raise RuntimeError(
            "indices.pkl must be dict or pandas Series-like (with .items())"
        )


def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404, detail=f"Title not found in local dataset: '{title}'"
    )


def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Returns list of (title, score) from local df using cosine similarity on TF-IDF matrix.
    Safe against missing columns/rows.
    """
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx = get_local_idx_by_title(query_title)

    # query vector
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()

    # sort descending
    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    """
    Uses TMDB search by title to fetch poster for a local title.
    If not found, returns None (never crashes the endpoint).
    """
    try:
        m = await tmdb_search_first(title)
        if not m:
            local_row = local_find_best_movie(title)
            return local_row_to_card(local_row) if local_row is not None else None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except HTTPException as e:
        if not tmdb_error_fallback_ok(e):
            return None
    except Exception:
        return None

    local_row = local_find_best_movie(title)
    return local_row_to_card(local_row) if local_row is not None else None


# =========================
# STARTUP: LOAD PICKLES
# =========================
@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX
    global metadata_df, metadata_by_id

    # Load df
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    # Load indices
    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    # Load TF-IDF matrix (usually scipy sparse)
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    # Load tfidf vectorizer (optional, not used directly here)
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    # Build normalized map
    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)
    metadata_df = build_local_metadata_catalog()
    metadata_by_id = (
        metadata_df.set_index("tmdb_id", drop=False)
        if metadata_df is not None and not metadata_df.empty
        else pd.DataFrame()
    )

    # sanity
    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")


# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- HOME FEED (TMDB) ----------
@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    """
    Home feed for Streamlit (posters).
    category:
      - trending (trending/movie/day)
      - popular, top_rated, upcoming, now_playing  (movie/{category})
    """
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
            return await tmdb_cards_from_results(data.get("results", []), limit=limit)

        if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
            raise HTTPException(status_code=400, detail="Invalid category")

        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)

    except HTTPException as e:
        if tmdb_error_fallback_ok(e):
            cards = local_home_cards(category=category, limit=limit)
            if cards:
                return cards
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")


# ---------- TMDB KEYWORD SEARCH (MULTIPLE RESULTS) ----------
@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    """
    Returns RAW TMDB shape with 'results' list.
    Streamlit will use it for:
      - dropdown suggestions
      - grid results
    """
    try:
        return await tmdb_search_movies(query=query, page=page)
    except HTTPException as e:
        if tmdb_error_fallback_ok(e):
            return local_search_movies(query=query, page=page)
        raise


# ---------- MOVIE DETAILS (SAFE ROUTE) ----------
@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    try:
        return await tmdb_movie_details(tmdb_id)
    except HTTPException as e:
        if tmdb_error_fallback_ok(e):
            local_row = get_local_movie_row(tmdb_id)
            if local_row is not None:
                return local_row_to_details(local_row)
        raise


# ---------- GENRE RECOMMENDATIONS ----------
@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    """
    Given a TMDB movie ID:
    - fetch details
    - pick first genre
    - discover movies in that genre (popular)
    """
    try:
        details = await tmdb_movie_details(tmdb_id)
        if not details.genres:
            return []

        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )
        cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
        return [c for c in cards if c.tmdb_id != tmdb_id]
    except HTTPException as e:
        if tmdb_error_fallback_ok(e):
            return local_genre_recommendations(
                source_row=get_local_movie_row(tmdb_id),
                limit=limit,
                exclude_tmdb_id=tmdb_id,
            )
        raise


# ---------- TF-IDF ONLY (debug/useful) ----------
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


# ---------- BUNDLE: Details + TF-IDF recs + Genre recs ----------
@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    """
    This endpoint is for when you have a selected movie and want:
      - movie details
      - TF-IDF recommendations (local) + posters
      - Genre recommendations (TMDB) + posters

    NOTE:
    - It selects the BEST match from TMDB for the given query.
    - If you want MULTIPLE matches, use /tmdb/search
    """
    details: Optional[TMDBMovieDetails] = None

    try:
        best = await tmdb_search_first(query)
        if best:
            details = await tmdb_movie_details(int(best["id"]))
    except HTTPException as e:
        if not tmdb_error_fallback_ok(e):
            raise

    local_row = None
    if details is None:
        local_row = local_find_best_movie(query)
        if local_row is None:
            raise HTTPException(status_code=404, detail=f"No movie found for query: {query}")
        details = local_row_to_details(local_row)
    else:
        local_row = get_local_movie_row(details.tmdb_id)

    # 1) TF-IDF recommendations (never crash endpoint)
    tfidf_items: List[TFIDFRecItem] = []

    recs: List[Tuple[str, float]] = []
    try:
        # try local dataset by TMDB title
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        # fallback to user query
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    for title, score in recs:
        card = await attach_tmdb_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=score, tmdb=card))

    # 2) Genre recommendations (TMDB discover by first genre)
    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        try:
            genre_id = details.genres[0]["id"]
            discover = await tmdb_get(
                "/discover/movie",
                {
                    "with_genres": genre_id,
                    "language": "en-US",
                    "sort_by": "popularity.desc",
                    "page": 1,
                },
            )
            cards = await tmdb_cards_from_results(
                discover.get("results", []), limit=genre_limit
            )
            genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]
        except HTTPException as e:
            if not tmdb_error_fallback_ok(e):
                raise
            genre_recs = local_genre_recommendations(
                source_row=local_row,
                limit=genre_limit,
                exclude_tmdb_id=details.tmdb_id,
            )

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )

@app.get("/movie/reviews/{tmdb_id}")
async def movie_reviews(tmdb_id: int):
    try:
        data = await tmdb_get(f"/movie/{tmdb_id}/reviews", {"language": "en-US"})
    except HTTPException as e:
        if tmdb_error_fallback_ok(e):
            return []
        raise
    
    reviews = []
    for r in data.get("results", [])[:5]:  # limit to 5
        reviews.append({
            "author": r.get("author"),
            "content": r.get("content"),
            "rating": r.get("author_details", {}).get("rating")
        })
    
    return reviews
