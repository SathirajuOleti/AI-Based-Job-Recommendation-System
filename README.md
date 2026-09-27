# AI-Based Job Recommendation System

An intelligent, explainable, and full-stack web application that analyzes candidate profiles and recommends the top 5 most relevant jobs using Natural Language Processing (NLP), TF-IDF Vectorization, and Cosine Similarity.

Developed as a **B.Tech Final-Year Capstone Project / Professional Portfolio Project**.

---

## Table of Contents
- [1. Project Overview](#1-project-overview)
- [2. Key Features](#2-key-features)
- [3. Technologies Used](#3-technologies-used)
- [4. Project Architecture & Workflow](#4-project-architecture--workflow)
- [5. Project Folder Structure](#5-project-folder-structure)
- [6. Dataset Details](#6-dataset-details)
- [7. Machine Learning & Recommendation Algorithm](#7-machine-learning--recommendation-algorithm)
  - [TF-IDF Vectorization](#tf-idf-vectorization)
  - [Cosine Similarity](#cosine-similarity)
  - [Multi-Factor Scoring Formula](#multi-factor-scoring-formula)
- [8. Installation & Setup Guide](#8-installation--setup-guide)
  - [Prerequisites](#prerequisites)
  - [Step 1: Clone or Navigate to Project](#step-1-clone-or-navigate-to-project)
  - [Step 2: Create a Virtual Environment](#step-2-create-a-virtual-environment)
  - [Step 3: Install Dependencies](#step-3-install-dependencies)
  - [Step 4: Initialize the Database](#step-4-initialize-the-database)
  - [Step 5: Run the Web Application](#step-5-run-the-web-application)
- [9. Example Candidate Input & Expected Output](#9-example-candidate-input--expected-output)
- [10. Error Handling & Form Validation](#10-error-handling--form-validation)
- [11. Interview & Viva Questions with Answers](#11-interview--viva-questions-with-answers)
- [12. Future Enhancements](#12-future-enhancements)

---

## 1. Project Overview

Finding the right job in the software and data technology domain requires matching multiple dimensions of a candidate's background—such as technical skills, qualification, years of experience, and role preferences—against complex job specifications. Traditional job portals frequently rely on rigid keyword lookups or manual boolean filters that fail to capture semantic relationships.

The **AI-Based Job Recommendation System** bridges this gap. It treats candidate profiles and job descriptions as unstructured text documents, maps them into a high-dimensional vector space using **TF-IDF (Term Frequency - Inverse Document Frequency)**, calculates mathematical alignment via **Cosine Similarity**, and combines this with multi-factor scoring (job role, location, and experience) to return the **top 5 ranked job recommendations**.

> **Note on Model Type:** This project is intentionally built as an **NLP / Machine Learning content-based recommendation system**. It does not use generative AI or deep neural networks, making every recommendation 100% mathematically transparent, fast, and explainable in academic viva examinations.

---

## 2. Key Features

- **Candidate Profiling Form:** Accepts skills, qualification, years of experience, domain interests, preferred role, and preferred location.
- **Preloaded Quick-Test Profiles:** 1-click test buttons for Data Analyst, Python Developer, ML Engineer, and Web Developer to facilitate live demonstrations.
- **Realistic Job Dataset:** 115+ realistic job records covering diverse IT sectors (Software Engineering, Data Science, AI/ML, DevOps, Cloud, Web, QA).
- **SQLite Relational Persistence:** Serverless, zero-config relational storage populated automatically from CSV.
- **Modular NLP Preprocessing:** Stop-word removal, lowercase normalization, punctuation replacement, and weighted `combined_text` feature creation.
- **Scikit-learn Vectorization:** Sublinear TF-IDF scaling and unigram + bigram feature extraction.
- **Multi-Factor Ranking:** 70% TF-IDF Cosine Similarity + 15% Role Match + 10% Location Match + 5% Experience Match.
- **Explainable Results UI:** Every recommended job includes a percentage match score alongside an expandable score breakdown accordion.
- **Robust Validation & Error Handling:** Graceful input validation (no empty fields, non-negative numeric experience) with friendly error alerts.

---

## 3. Technologies Used

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.9+ | Core programming language |
| **Backend Framework** | Flask | Lightweight WSGI web framework, routing, and controller logic |
| **Data Processing** | Pandas, NumPy | Dataset ingestion, data cleaning, text concatenation |
| **Machine Learning** | Scikit-learn | `TfidfVectorizer`, `cosine_similarity` |
| **Database** | SQLite3 | Embedded, zero-configuration relational database |
| **Frontend UI** | HTML5, CSS3, Bootstrap 5 | Modern, responsive, and mobile-friendly user interface |
| **Icons & Typography** | Bootstrap Icons, Plus Jakarta Sans | Visual iconography and clean typography |
| **Templating** | Jinja2 | Dynamic server-side HTML rendering |

---

## 4. Project Architecture & Workflow

The end-to-end data pipeline and execution flow:

```text
[ Candidate / User ]
        │
        ▼ (Inputs: Skills, Role, Exp, Location, Qualification, Interests)
[ HTML5 / Bootstrap 5 Form ] (index.html)
        │
        ▼ (HTTP POST /recommend)
[ Flask Controller ] (app.py)
   ├── Input Validation & Sanitization
   └── Calls Recommender Engine
        │
        ▼
[ Data Preprocessing ] (data_preprocessing.py)
   ├── Normalizes text (lowercase, regex cleanup, strip whitespace)
   ├── Formulates candidate profile string
   └── Prepares job corpus 'combined_text'
        │
        ▼
[ TF-IDF Vectorizer ] (Scikit-learn)
   ├── Fits & transforms job corpus into TF-IDF Matrix (N x M)
   └── Transforms candidate profile into Candidate Vector (1 x M)
        │
        ▼
[ Cosine Similarity ] (sklearn.metrics.pairwise)
   └── Calculates similarity between candidate vector and every job vector
        │
        ▼
[ Multi-Factor Scoring Engine ] (recommender.py)
   ├── 70% Content Similarity (Cosine Similarity)
   ├── 15% Preferred Role Match
   ├── 10% Preferred Location Match
   └──  5% Experience Compatibility Match
        │
        ▼
[ Ranking & Top-5 Selection ]
   └── Sorts jobs by Final Score in descending order
        │
        ▼ (Passes top 5 matches to Jinja2 template)
[ Results Page ] (results.html)
```

---

## 5. Project Folder Structure

```text
AI-Job-Recommendation-System/
│
├── app.py                   # Main Flask application with routes and controllers
├── recommender.py           # ML recommendation engine (TF-IDF, Cosine Similarity, Scoring)
├── data_preprocessing.py    # Text cleaning, normalization, and profile string builder
├── database.py              # SQLite database creation, ingestion, and query helpers
├── requirements.txt         # Minimal Python package dependencies
├── README.md                # Comprehensive project documentation
│
├── data/
│   └── jobs.csv             # Realistic dataset containing 115+ curated job records
│
├── database/
│   └── jobs.db              # SQLite database file (created automatically on startup)
│
├── templates/
│   ├── base.html            # Base Jinja2 layout with navigation, modals, and footer
│   ├── index.html           # Candidate profile input form with quick-fill demos
│   ├── results.html         # Top 5 recommended jobs display with match scores
│   └── error.html           # Friendly error pages for 404, 500, or invalid input
│
└── static/
    └── css/
        └── style.css        # Custom styles, badges, and responsive UI polish
```

### Purpose of Every File
- **`app.py`**: Initializes the Flask app, handles routing for `/` and `/recommend`, performs input validation, and renders templates.
- **`recommender.py`**: Contains the core recommendation logic. Vectorizes data using Scikit-learn, computes cosine similarity, calculates multi-factor scores, and ranks jobs.
- **`data_preprocessing.py`**: Cleans textual inputs, handles null values, converts strings to lowercase, and creates the unified `combined_text` column.
- **`database.py`**: Manages the SQLite database schema and automatically loads `jobs.csv` into `jobs.db`.
- **`data/jobs.csv`**: The source dataset containing realistic job titles, required skills, companies, locations, experience requirements, and descriptions.
- **`templates/`**: Jinja2 HTML templates adhering to Bootstrap 5 responsive layout standards.
- **`static/css/style.css`**: Provides custom color styling, card elevation effects, and progress bars.

---

## 6. Dataset Details

The dataset `data/jobs.csv` consists of **115+ verified, internally consistent job postings** across major IT companies (Infosys, TCS, Wipro, Google, Microsoft, Amazon, Deloitte, Swiggy, Razorpay, etc.).

Each record includes:
- `job_id`: Unique integer identifier.
- `job_title`: Designation (e.g. *Python Developer*, *Data Analyst*, *Machine Learning Engineer*).
- `company`: Employing organization.
- `location`: Working city or *Remote*.
- `required_skills`: Comma-separated technical tools, frameworks, and programming languages.
- `qualification`: Required academic degree (*B.Tech*, *B.E.*, *M.Tech*, *MCA*, *BCA*, *B.Sc*, *Any Graduate*).
- `experience`: Required experience in years (0 to 3).
- `job_type`: Employment nature (*Full-time*, *Remote*, *Internship*).
- `description`: Detailed technical responsibilities and project focus.

---

## 7. Machine Learning & Recommendation Algorithm

### TF-IDF Vectorization
**TF-IDF** stands for **Term Frequency - Inverse Document Frequency**. It converts text strings into numerical vectors based on word importance:

1. **Term Frequency (TF):** Measures how frequently a word $t$ appears in document $d$:
   $$\text{TF}(t, d) = \frac{\text{Count of } t \text{ in } d}{\text{Total words in } d}$$

2. **Inverse Document Frequency (IDF):** Measures how unique or discriminative a word is across all $N$ documents:
   $$\text{IDF}(t, D) = \log\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$
   *If a common word like "developer" appears in every document, its IDF is low. If a specific skill like "TensorFlow" appears in only a few jobs, its IDF is high.*

3. **TF-IDF Score:**
   $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

### Cosine Similarity
Once the candidate profile and job descriptions are transformed into vectors in an $M$-dimensional space, **Cosine Similarity** calculates the cosine of the angle $\theta$ between the candidate vector $\vec{A}$ and the job vector $\vec{B}$:

$$\text{Cosine Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

- **Score of 1.0 (100%):** Identical orientation (exact conceptual match).
- **Score of 0.0 (0%):** Orthogonal vectors (no common features).

### Multi-Factor Scoring Formula
To balance pure text matching with real-world hiring criteria, the system applies an explainable weighted linear model:

$$\text{Final Score} = (0.70 \times S_{\text{content}}) + (0.15 \times S_{\text{role}}) + (0.10 \times S_{\text{location}}) + (0.05 \times S_{\text{experience}})$$

Where:
1. **$S_{\text{content}}$ (70%):** Cosine similarity between candidate TF-IDF vector and job's `combined_text` vector ($0.0 \le S \le 1.0$).
2. **$S_{\text{role}}$ (15%):**
   - Exact match or full substring match between preferred role and job title: $1.0$
   - Partial token overlap (e.g. sharing "developer"): $0.6$
   - No overlap: $0.0$
3. **$S_{\text{location}}$ (10%):**
   - Candidate city matches job location (or job is Remote): $1.0$
   - Different city: $0.0$
4. **$S_{\text{experience}}$ (5%):**
   - Candidate experience $\ge$ required experience: $1.0$
   - Candidate experience is 1 year below requirement: $0.7$
   - Candidate experience is 2 years below requirement: $0.4$
   - Otherwise: $0.1$

---

## 8. Installation & Setup Guide

### Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12 installed on your system.
- Git (optional, for cloning).

### Step 1: Navigate to Project Directory
Open your terminal (PowerShell, Command Prompt, or Bash) and navigate to the project directory:
```bash
cd "AI-Based Job Recommendation System"
```

### Step 2: Create a Virtual Environment
Isolate project dependencies by creating a Python virtual environment:

**On Windows (PowerShell/CMD):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
Install the required libraries using pip:
```bash
pip install -r requirements.txt
```

### Step 4: Initialize the Database (Optional / Automatic)
The database automatically initializes upon launching the Flask app. You can also run the initialization script directly:
```bash
python database.py
```
*Output: `Successfully inserted 115 job records into SQLite database.`*

### Step 5: Run the Web Application
Start the Flask development server:
```bash
python app.py
```

Open your web browser and visit:
```text
http://127.0.0.1:5000
```

---

## 9. Example Candidate Input & Expected Output

### Input Example
- **Skills:** `Python, Pandas, SQL, Machine Learning`
- **Qualification:** `B.Tech`
- **Years of Experience:** `1`
- **Interests:** `Data Analysis, Machine Learning`
- **Preferred Role:** `Data Analyst`
- **Preferred Location:** `Hyderabad`

### Expected Output
The system computes similarities across all 115 jobs and renders the top 5 matches:

1. **Data Analyst** at **Deloitte** (Hyderabad) — **Match Score: ~88-92%**
   - *High content similarity on Python, Pandas, SQL; exact role match; exact location match; meets experience.*
2. **Data Analyst** at **Amazon** (Hyderabad) — **Match Score: ~75-80%**
   - *High skills overlap; exact role and location match.*
3. **Data Scientist** at **TCS** (Hyderabad) — **Match Score: ~70-75%**
   - *Overlaps on Python, Machine Learning, Hyderabad location.*
4. **Python Developer** at **Infosys** (Hyderabad) — **Match Score: ~68-72%**
   - *Overlaps on Python, SQL, Hyderabad location.*
5. **Data Analyst** at **AnalyticsHQ** (Remote) — **Match Score: ~67-71%**
   - *Overlaps on Pandas, SQL, Data Analyst role, Remote accessibility.*

---

## 10. Error Handling & Form Validation

- **Missing Fields:** Clear warnings notifying the user if mandatory inputs like skills or preferred role are left blank.
- **Experience Bounds:** Validates that years of experience is a non-negative number between 0 and 50.
- **Data Retention:** Preserves entered values on validation failure so the candidate does not have to retype.
- **Database Safety:** Safe parameterized SQL queries (`?`) prevent SQL injection attacks.
- **Custom Error Views:** Graceful 404 (Not Found) and 500 (Internal Server Error) templates without exposing Python stack traces to end-users.

---

## 11. Interview & Viva Questions with Answers

#### Q1: Why did you choose TF-IDF and Cosine Similarity over simple keyword matching?
> **Answer:** Simple keyword matching (like `if 'python' in skills`) only checks binary presence and cannot evaluate degree of relevance or relative importance. TF-IDF penalizes commonly occurring words while elevating domain-specific skills, and Cosine Similarity measures the angular distance between vectors, allowing us to quantify subtle semantic relevance independent of document length.

#### Q2: Why is Cosine Similarity preferred over Euclidean Distance for text recommendation?
> **Answer:** Euclidean distance is sensitive to document length (a longer job description with more repeated words will have a larger Euclidean distance even if it discusses the exact same topics). Cosine similarity measures the angle between vectors rather than magnitude, making it invariant to text length.

#### Q3: How does the system handle a fresher with 0 years of experience?
> **Answer:** In our multi-factor scoring formula, when a candidate enters `0` and a job requires `0` (entry-level / fresher), the experience match score evaluates to `1.0` (100%). For jobs requiring 1 year, the penalty is small (`0.7`), allowing qualified freshers to still be matched based on high technical skill similarity.

#### Q4: Why did you use SQLite instead of MySQL or PostgreSQL?
> **Answer:** SQLite is serverless, zero-configuration, and stores the entire database in a single portable file (`database/jobs.db`). It natively integrates with Python standard library (`sqlite3`), eliminating external service setup for examiners or interviewers evaluating the project.

#### Q5: Is this system considered Generative AI?
> **Answer:** No. It is an information retrieval and Machine Learning / NLP recommendation system. Accurately characterizing it as a Content-Based Recommender using TF-IDF and Vector Space Modeling demonstrates solid understanding of classical ML principles.

---

## 12. Future Enhancements

1. **Resume PDF Parsing:** Allow candidates to upload their PDF/DOCX resumes and extract skills using spaCy / regex.
2. **Collaborative Filtering:** Combine content-based matching with collaborative filtering based on historical user clicks and applications.
3. **Dense Vector Embeddings:** Compare TF-IDF against semantic transformer embeddings (such as Sentence-BERT or Word2Vec).
4. **Recruiter Portal:** Provide an administrative dashboard for employers to post new jobs directly into the database.
