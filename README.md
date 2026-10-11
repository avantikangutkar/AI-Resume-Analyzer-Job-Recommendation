# AI Resume Analyzer and Job Recommendation System

## Project Overview
This project uses Natural Language Processing techniques,
skill extraction, and TF-IDF text similarity to analyze resumes
and recommend relevant job roles.

## Features
- Upload PDF and DOCX resumes
- Extract and clean resume text
- Identify technical skills
- Recommend the top three job roles
- Calculate skill coverage and text similarity
- Identify missing skills for a selected role
- Display skill coverage charts

## Technologies Used
- Python
- Streamlit
- Pandas and NumPy
- pypdf and python-docx
- Scikit-learn
- TF-IDF and cosine similarity
- Matplotlib

## Run Locally
1. Install Python.
2. Install dependencies:
   `python -m pip install -r requirements.txt`
3. Start the application:
   `python -m streamlit run app.py`

## Limitations
- Recommendations depend on a predefined set of job roles
  and technical skills.
- Text similarity and skill coverage are estimates, not
  guarantees of job suitability.
- Scanned resumes may require OCR support.
- A missing keyword does not necessarily mean a person
  lacks that skill.

## Responsible Use
This application is intended for career guidance and learning.
It should not be used as the sole basis for hiring decisions.
Resume data should be handled securely.
