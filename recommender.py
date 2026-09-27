"""
recommender.py
--------------
Core Machine Learning Recommendation Engine using NLP, TF-IDF Vectorization,
and Cosine Similarity.

How the Recommendation Engine Works:
1. TF-IDF (Term Frequency - Inverse Document Frequency):
   - Converts the text descriptions and skills of all jobs into numerical feature vectors.
   - Words that appear frequently in a specific job (like 'Python', 'Pandas') receive higher weights,
     while common words (stop words like 'and', 'the') receive lower weights.
2. Candidate Vectorization:
   - The candidate's skills, qualifications, preferred role, and interests are transformed
     into the same TF-IDF vector space.
3. Cosine Similarity:
   - Measures the cosine of the angle between the candidate vector and each job vector.
   - A cosine similarity of 1.0 means an exact conceptual alignment, while 0.0 means no overlap.
4. Explainable Multi-Factor Scoring Formula:
   Final Score (100%) =
       70% Content Similarity (TF-IDF Cosine Similarity)
     + 15% Preferred Role Match
     + 10% Preferred Location Match
     +  5% Experience Compatibility Match
5. Ranking:
   - Jobs are sorted by Final Score in descending order.
   - The top 5 most relevant jobs are returned to the user with percentage match scores.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from database import get_all_jobs
from data_preprocessing import preprocess_jobs_dataframe, create_candidate_profile, clean_text


def compute_role_match(candidate_role: str, job_title: str) -> float:
    """
    Computes a normalized score (0.0 to 1.0) indicating how well the candidate's
    preferred role matches the job title.
    - Exact match or full substring match: 1.0
    - Partial token overlap (e.g., both contain 'developer'): 0.6
    - No match: 0.0
    """
    cand = clean_text(candidate_role)
    title = clean_text(job_title)

    if not cand or not title:
        return 0.0

    if cand == title or cand in title or title in cand:
        return 1.0

    cand_tokens = set(cand.split())
    title_tokens = set(title.split())
    overlap = cand_tokens.intersection(title_tokens)

    if overlap:
        return 0.6

    return 0.0


def compute_location_match(candidate_location: str, job_location: str) -> float:
    """
    Computes a normalized score (0.0 to 1.0) for location compatibility.
    - Exact match (case-insensitive): 1.0
    - Job is 'Remote' or Candidate preferred 'Remote': 1.0
    - No match: 0.0
    """
    cand = clean_text(candidate_location)
    loc = clean_text(job_location)

    if not cand or not loc:
        return 0.0

    if cand == loc:
        return 1.0

    if "remote" in cand or "remote" in loc:
        return 1.0

    return 0.0


def compute_experience_match(candidate_exp: float, job_exp: int) -> float:
    """
    Computes an experience compatibility score (0.0 to 1.0).
    - If candidate meets or exceeds job experience requirement: 1.0
    - If candidate has 1 year less than required: 0.7
    - If candidate has 2 years less than required: 0.4
    - Otherwise: 0.1
    """
    try:
        diff = candidate_exp - job_exp
        if diff >= 0:
            return 1.0
        elif diff >= -1:
            return 0.7
        elif diff >= -2:
            return 0.4
        else:
            return 0.1
    except Exception:
        return 0.5


def recommend_jobs(
    candidate_skills: str,
    candidate_qualification: str,
    candidate_experience: float,
    candidate_interests: str,
    candidate_role: str,
    candidate_location: str,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Recommends the top N jobs for a given candidate profile using TF-IDF vectorization,
    cosine similarity, and multi-factor scoring.

    Returns:
        List of job dictionaries containing details and match score percentages.
    """
    # 1. Fetch raw job records from SQLite
    raw_jobs_df = get_all_jobs()
    if raw_jobs_df.empty:
        return []

    # 2. Preprocess jobs dataframe and create 'combined_text'
    jobs_df = preprocess_jobs_dataframe(raw_jobs_df)

    # 3. Create candidate profile text
    candidate_profile = create_candidate_profile(
        skills=candidate_skills,
        qualification=candidate_qualification,
        experience=candidate_experience,
        interests=candidate_interests,
        preferred_role=candidate_role,
        preferred_location=candidate_location
    )

    # 4. Initialize and fit TF-IDF Vectorizer
    # stop_words='english' filters out non-informative English articles/prepositions
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    
    # Fit vectorizer on all job combined texts
    job_tfidf_matrix = vectorizer.fit_transform(jobs_df["combined_text"])

    # Transform candidate profile into the exact same vector space
    candidate_vector = vectorizer.transform([candidate_profile])

    # 5. Compute Cosine Similarity between candidate vector and all jobs
    cosine_sim_scores = cosine_similarity(candidate_vector, job_tfidf_matrix).flatten()

    # 6. Calculate Final Multi-Factor Recommendation Score for each job
    # Formula:
    # 70% Content Similarity + 15% Role Match + 10% Location Match + 5% Experience Match
    scored_jobs = []
    for idx, row in jobs_df.iterrows():
        content_sim = float(cosine_sim_scores[idx])
        role_match = compute_role_match(candidate_role, row["job_title"])
        loc_match = compute_location_match(candidate_location, row["location"])
        exp_match = compute_experience_match(candidate_experience, int(row["experience"]))

        # Weighted final score calculation
        final_score = (
            (0.70 * content_sim) +
            (0.15 * role_match) +
            (0.10 * loc_match) +
            (0.05 * exp_match)
        )

        # Convert to percentage
        match_percentage = round(final_score * 100, 1)

        job_info = {
            "job_id": int(row["job_id"]),
            "job_title": str(row["job_title"]),
            "company": str(row["company"]),
            "location": str(row["location"]),
            "job_type": str(row["job_type"]),
            "required_skills": str(row["required_skills"]),
            "qualification": str(row["qualification"]),
            "experience": int(row["experience"]),
            "description": str(row["description"]),
            "match_score": match_percentage,
            # Component scores for transparency and viva demonstration
            "content_score": round(content_sim * 100, 1),
            "role_score": round(role_match * 100, 1),
            "location_score": round(loc_match * 100, 1),
            "experience_score": round(exp_match * 100, 1)
        }
        scored_jobs.append(job_info)

    # 7. Sort jobs by match_score descending
    scored_jobs.sort(key=lambda x: x["match_score"], reverse=True)

    # 8. Return top N recommendations
    return scored_jobs[:top_n]


if __name__ == "__main__":
    print("Testing recommendation engine...")
    test_recommendations = recommend_jobs(
        candidate_skills="Python, Pandas, SQL, Machine Learning",
        candidate_qualification="B.Tech",
        candidate_experience=1,
        candidate_interests="Data Analysis, Machine Learning",
        candidate_role="Data Analyst",
        candidate_location="Hyderabad",
        top_n=5
    )
    print(f"\nTop {len(test_recommendations)} Job Recommendations:")
    for i, job in enumerate(test_recommendations, 1):
        print(f"{i}. {job['job_title']} at {job['company']} ({job['location']}) - Match: {job['match_score']}%")
