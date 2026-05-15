import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import GenerativeModel

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Check API key existence
if not API_KEY:
    raise ValueError("❌ GEMINI_API_KEY is missing. Check your .env file.")

# Configure Gemini
genai.configure(api_key=API_KEY)
model = GenerativeModel(model_name="models/gemini-1.5-flash-002")

def generate_cover_letter(resume_text, job_role):
    """
    Generates a professional cover letter using Gemini AI based on resume content and job role.
    """
    prompt = f"""
    You are a professional career assistant.
    
    Write a formal, concise, and effective cover letter for the job role of **{job_role}**.
    Use the following resume text as context:

    ---
    {resume_text}
    ---

    Structure:
    1. Polite and brief introduction
    2. Highlight most relevant skills and experiences for {job_role}
    3. Show enthusiasm and cultural fit
    4. Professional closing
    """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error generating cover letter: {e}"
