# 🎬 Movie Recommendation System

> AI-powered movie recommendation system using NLP (TF-IDF) with FastAPI backend and Streamlit frontend.

🔗 **Live App:** https://movierecommenderappli.streamlit.app/

---

## 🚀 Overview

This project solves the problem of **content overload** by recommending movies based on user preferences.

It uses:

* **TF-IDF (NLP)** for content-based filtering
* **TMDB API** for real-time movie data (posters, ratings, details)
* **FastAPI** for backend
* **Streamlit** for frontend UI

---

## ✨ Features

* 🔍 Search movies with suggestions
* 🎯 Personalized recommendations (TF-IDF)
* 🎭 Genre-based recommendations
* 🖼 Movie posters & details using TMDB API
* ⚡ Fast and interactive UI

---

## 🛠 Tech Stack

* Python
* FastAPI
* Streamlit
* Pandas, NumPy
* Scikit-learn (TF-IDF)
* TMDB API

---

## 📂 Project Structure

```
movie-rec/
│── app.py                # Streamlit frontend
│── main.py               # FastAPI backend
│── requirements.txt
│── df.pkl
│── tfidf.pkl
│── tfidf_matrix.pkl
│── indices.pkl
│── movies_metadata.csv
│── README.md
```

---

## ⚙️ Setup & Run Locally

### 1️⃣ Clone Repository

```bash
git clone https://github.com/devangyadav/movierecommender.git
cd movierecommender
```

---

### 2️⃣ Create Virtual Environment

#### Mac / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Get TMDB API Key 🔑

1. Go to: https://www.themoviedb.org/
2. Create account → Settings → API
3. Generate API key

---

### 5️⃣ Add API in `.env` file

```
.env
```

Add:

```
TMDB_API_KEY=your_api_key_here
```

---

### 6️⃣ Run Backend (FastAPI)

```bash
uvicorn main:app --reload
```

👉 Runs at:

```
http://127.0.0.1:8000
```

---

### 7️⃣ Run Frontend (Streamlit)

Open new terminal:

```bash
streamlit run app.py
```

👉 Runs at:

```
http://localhost:8501
```

---

## 🌐 Deployment

* Backend deployed on Render
* Frontend deployed on Streamlit Cloud

🔗 Live App:
https://movierecommenderappli.streamlit.app/

## 👨‍💻 Author

**Devang Yadav**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
