"""
app.py
------
Flask Web Application for the AI-Based Job Recommendation System.

Routes:
- GET  /          : Renders the homepage with the candidate profile form.
- POST /recommend : Validates candidate input, runs the recommendation engine,
                    and renders the top 5 recommended jobs.
- Error Handlers  : Gracefully catches 404, 500, and unexpected application errors.
"""

import os
from flask import Flask, render_template, request, redirect, url_for, flash
from recommender import recommend_jobs
from database import init_db

app = Flask(__name__)
# Secret key used for secure session handling and flash messages
app.secret_key = os.environ.get("SECRET_KEY", "ai_job_recommender_secret_key_2026")


@app.before_request
def setup_database_once():
    """
    Ensures that the SQLite database is initialized with job records before
    handling any incoming requests.
    """
    if not hasattr(app, "_db_initialized"):
        try:
            init_db()
            app._db_initialized = True
        except Exception as e:
            app.logger.error(f"Failed to auto-initialize database: {e}")


@app.route("/", methods=["GET"])
def index():
    """
    Renders the homepage containing the candidate profile input form.
    """
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Handles form submission from the candidate:
    1. Extracts and sanitizes input data.
    2. Validates mandatory fields and data types.
    3. Calls recommender.py to obtain top 5 job matches.
    4. Renders results.html with recommended jobs and match scores.
    """
    try:
        # Retrieve form data
        skills = request.form.get("skills", "").strip()
        qualification = request.form.get("qualification", "").strip()
        experience_raw = request.form.get("experience", "").strip()
        interests = request.form.get("interests", "").strip()
        preferred_role = request.form.get("preferred_role", "").strip()
        preferred_location = request.form.get("preferred_location", "").strip()

        # Preserve input in a dictionary so the user doesn't lose data if validation fails
        form_data = {
            "skills": skills,
            "qualification": qualification,
            "experience": experience_raw,
            "interests": interests,
            "preferred_role": preferred_role,
            "preferred_location": preferred_location,
        }

        # ----------------- Form Validation -----------------
        if not skills:
            flash("Please enter your skills (e.g. Python, SQL, Machine Learning).", "warning")
            return render_template("index.html", form_data=form_data)

        if not qualification:
            flash("Please select or enter your highest qualification.", "warning")
            return render_template("index.html", form_data=form_data)

        if not experience_raw:
            flash("Please enter your years of experience.", "warning")
            return render_template("index.html", form_data=form_data)

        # Validate numeric experience
        try:
            experience = float(experience_raw)
            if experience < 0 or experience > 50:
                flash("Please enter a valid number of years of experience (between 0 and 50).", "warning")
                return render_template("index.html", form_data=form_data)
        except ValueError:
            flash("Please enter a valid number for years of experience (e.g., 0, 1, 2).", "danger")
            return render_template("index.html", form_data=form_data)

        if not preferred_role:
            flash("Please enter your preferred job role.", "warning")
            return render_template("index.html", form_data=form_data)

        if not preferred_location:
            flash("Please enter your preferred location.", "warning")
            return render_template("index.html", form_data=form_data)

        # ----------------- Recommendation Processing -----------------
        recommendations = recommend_jobs(
            candidate_skills=skills,
            candidate_qualification=qualification,
            candidate_experience=experience,
            candidate_interests=interests,
            candidate_role=preferred_role,
            candidate_location=preferred_location,
            top_n=5
        )

        if not recommendations:
            return render_template(
                "error.html",
                error_title="No Matches Found",
                error_message="We could not find any jobs matching your profile in our current database. Please try adjusting your skills or preferred role."
            )

        candidate_summary = {
            "skills": skills,
            "qualification": qualification,
            "experience": int(experience) if experience.is_integer() else experience,
            "interests": interests,
            "preferred_role": preferred_role,
            "preferred_location": preferred_location,
        }

        return render_template(
            "results.html",
            recommendations=recommendations,
            candidate=candidate_summary
        )

    except Exception as e:
        app.logger.error(f"Error generating recommendations: {e}")
        return render_template(
            "error.html",
            error_title="Recommendation Error",
            error_message="An unexpected error occurred while analyzing your profile. Please check your inputs and try again."
        ), 500


@app.errorhandler(404)
def page_not_found(e):
    """
    Renders user-friendly 404 Page Not Found error.
    """
    return render_template(
        "error.html",
        error_title="Page Not Found (404)",
        error_message="The page you are looking for does not exist. Please return to the home page."
    ), 404


@app.errorhandler(500)
def internal_server_error(e):
    """
    Renders user-friendly 500 Internal Server Error.
    """
    return render_template(
        "error.html",
        error_title="Internal Server Error (500)",
        error_message="Something went wrong on our end. Please return to the homepage and try again."
    ), 500


if __name__ == "__main__":
    # Initialize the database on startup if running directly
    init_db()
    # Run Flask development server
    print("\nStarting AI-Based Job Recommendation System on http://127.0.0.1:5000\n")
    app.run(debug=True, host="127.0.0.1", port=5000)
