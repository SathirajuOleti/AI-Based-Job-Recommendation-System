"""
data_preprocessing.py
---------------------
Preprocesses the job dataset and candidate inputs using Pandas and standard NLP cleaning.

Key Steps:
1. Missing Value Handling: Fills missing or null fields with empty strings.
2. Text Normalization: Converts all text to lowercase to ensure case-insensitive matching.
3. Whitespace Cleaning: Trims leading, trailing, and redundant inner spaces.
4. Combined Text Generation: Combines job title, required skills, qualification, job type,
   and description into a unified 'combined_text' feature for TF-IDF vectorization.
5. Candidate Profile Construction: Combines candidate skills, qualification, interests,
   role, and experience into a clean text profile matching the feature space.
"""

import re
import pandas as pd


def clean_text(text: str) -> str:
    """
    Cleans a text string by:
    - Converting to string and handling None/NaN
    - Converting to lowercase
    - Replacing commas, slashes, and special characters with spaces
    - Collapsing multiple whitespace characters into a single space
    """
    if text is None or pd.isna(text):
        return ""
    
    text = str(text).lower()
    # Replace punctuation commonly found in skill lists with spaces
    text = re.sub(r"[,/|;()\-+]", " ", text)
    # Remove any extra non-alphanumeric noise while preserving words and numbers
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_jobs_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame of jobs and creates a preprocessed copy with:
    - Null values handled
    - Individual fields normalized
    - A 'combined_text' column created for TF-IDF feature extraction
    """
    if df is None or df.empty:
        return pd.DataFrame()

    processed_df = df.copy()

    # Fill any null values across all columns
    processed_df = processed_df.fillna("")

    # Clean text columns
    text_cols = ["job_title", "required_skills", "qualification", "job_type", "description"]
    for col in text_cols:
        if col in processed_df.columns:
            processed_df[f"cleaned_{col}"] = processed_df[col].apply(clean_text)
        else:
            processed_df[f"cleaned_{col}"] = ""

    # Combine relevant fields into a single text representation
    # Weighting: Repeating job_title and required_skills gives them higher term frequency
    processed_df["combined_text"] = (
        processed_df["cleaned_job_title"] + " " +
        processed_df["cleaned_job_title"] + " " +
        processed_df["cleaned_required_skills"] + " " +
        processed_df["cleaned_required_skills"] + " " +
        processed_df["cleaned_qualification"] + " " +
        processed_df["cleaned_job_type"] + " " +
        processed_df["cleaned_description"]
    ).str.strip()

    return processed_df


def create_candidate_profile(
    skills: str,
    qualification: str,
    experience: int or float,
    interests: str,
    preferred_role: str,
    preferred_location: str
) -> str:
    """
    Constructs a unified candidate profile text for vectorization.
    Emphasizes skills, preferred role, and interests to align with job descriptions.
    """
    cleaned_skills = clean_text(skills)
    cleaned_qualification = clean_text(qualification)
    cleaned_interests = clean_text(interests)
    cleaned_role = clean_text(preferred_role)
    cleaned_location = clean_text(preferred_location)

    # Build weighted candidate profile text
    # Skills and preferred role are repeated to boost importance in TF-IDF
    profile_components = [
        cleaned_role,
        cleaned_role,
        cleaned_skills,
        cleaned_skills,
        cleaned_interests,
        cleaned_qualification,
        f"experience {experience} years",
        cleaned_location
    ]

    return " ".join([c for c in profile_components if c]).strip()


if __name__ == "__main__":
    from database import get_all_jobs
    print("Testing data preprocessing...")
    df = get_all_jobs()
    processed = preprocess_jobs_dataframe(df)
    print(f"Total jobs processed: {len(processed)}")
    print("Sample combined text from Job #1:")
    print(processed.iloc[0]["combined_text"][:200] + "...")
