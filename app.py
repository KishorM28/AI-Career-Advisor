from dotenv import load_dotenv
import logging
import os
import pickle
import re
from io import BytesIO

import fitz  # PyMuPDF
from flask import Flask, render_template, request, session, send_file
from fpdf import FPDF
from groq import Groq

# Load environment variables from .env
load_dotenv()

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Use a persistent secret key from .env so sessions remain valid across restarts.
app.secret_key = os.getenv("FLASK_SECRET_KEY")
if not app.secret_key:
    raise RuntimeError(
        "FLASK_SECRET_KEY is missing. Generate a secret key and add it to your .env file."
    )

# Load the trained resume classification model and vectorizer.
with open("resume_classifier.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Groq API configuration.
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")

client = Groq(api_key=groq_api_key)
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


required_skills_by_role = {
    "AI Engineer": ["Python", "TensorFlow", "PyTorch", "Deep Learning", "ML"],
    "Data Scientist": ["Python", "Machine Learning", "Statistics", "Pandas", "SQL", "Scikit-learn", "TensorFlow"],
    "Web Developer": ["HTML", "CSS", "JavaScript", "React", "Node.js"],
    "Designer": ["Figma", "Adobe XD", "UI/UX", "Sketch", "Visual Design"],
    "Full Stack Developer": ["HTML", "CSS", "JavaScript", "React", "Node.js", "Express", "MongoDB"],
    "Backend Developer": ["Node.js", "Express", "MongoDB", "SQL", "APIs"],
    "Frontend Developer": ["HTML", "CSS", "JavaScript", "React", "UI/UX"],
    "DevOps Engineer": ["Docker", "Kubernetes", "CI/CD", "AWS", "Terraform"],
    "Mobile App Developer": ["Flutter", "React Native", "Java", "Kotlin", "Swift"],
    "Cloud Engineer": ["AWS", "Azure", "GCP", "Terraform", "Linux"],
    "ML Engineer": ["Python", "Scikit-learn", "Deep Learning", "TensorFlow", "MLOps"],
    "Database Admin": ["SQL", "MySQL", "PostgreSQL", "MongoDB", "Database Design"],
    "Graphic Designer": ["Adobe Photoshop", "Illustrator", "InDesign", "Sketch", "Figma", "Typography", "Branding", "UI/UX", "Video Editing"],
    "Musician": ["Music Theory", "Composition", "Sibelius", "Ableton", "Instrumental Skills", "Recording", "Sound Editing"],
    "Dancer": ["Choreography", "Stage Performance", "Classical Dance", "Hip-hop", "Contemporary", "Flexibility", "Rhythm"],
    "Actor": ["Stage Presence", "Improvisation", "Dialogue Delivery", "Script Reading", "Voice Modulation", "Screen Acting"],
    "Teacher": ["Classroom Management", "Lesson Planning", "Curriculum Design", "Educational Tools", "Assessment", "Communication"],
    "Business Analyst": ["Business Analysis", "Stakeholder Communication", "Requirement Gathering", "SQL", "Excel", "UML"],
    "Digital Marketer": ["SEO", "Google Analytics", "Social Media Marketing", "Email Marketing", "AdWords", "Content Strategy"],
    "Mechanical Engineer": ["CAD", "SolidWorks", "Thermodynamics", "MATLAB", "Mechanical Design"],
    "Civil Engineer": ["AutoCAD", "Structural Analysis", "Construction Management", "Site Supervision"],
    "Doctor": ["Diagnosis", "Patient Care", "Medical Knowledge", "Clinical Procedures", "EMR Systems"],
    "Nurse": ["Patient Monitoring", "Medication Administration", "Emergency Care", "Medical Records", "Compassionate Care"],
}

normalized_skill_map = {
    role.lower(): skills
    for role, skills in required_skills_by_role.items()
}

role_mapping = {
    role.lower(): role
    for role in required_skills_by_role
}
role_mapping.update({
    "data science": "Data Scientist",
    "ml engineer": "ML Engineer",
    "ai engineer": "AI Engineer",
    "fullstack developer": "Full Stack Developer",
    "full stack developer": "Full Stack Developer",
    "frontend developer": "Frontend Developer",
    "backend developer": "Backend Developer",
    "web developer": "Web Developer",
    "devops engineer": "DevOps Engineer",
    "cloud engineer": "Cloud Engineer",
    "business analyst": "Business Analyst",
    "digital marketer": "Digital Marketer",
    "graphic designer": "Graphic Designer",
    "mobile developer": "Mobile App Developer",
    "teacher": "Teacher",
    "actor": "Actor",
    "dancer": "Dancer",
    "doctor": "Doctor",
    "nurse": "Nurse",
    "designer": "Designer",
    "musician": "Musician",
    "mechanical engineer": "Mechanical Engineer",
    "civil engineer": "Civil Engineer",
})


def extract_text_from_pdf(file_storage):
    """Extract readable text from an uploaded PDF."""
    with fitz.open(stream=file_storage.read(), filetype="pdf") as document:
        return "\n".join(page.get_text() for page in document).strip()


def extract_skills(text):
    """Extract known skill phrases that appear in the resume."""
    text_lower = text.lower()
    found_skills = []

    for skills in required_skills_by_role.values():
        for skill in skills:
            # Match whole terms/phrases while allowing punctuation and whitespace.
            pattern = r"(?<!\w)" + re.escape(skill.lower()).replace(r"\ ", r"\s+") + r"(?!\w)"
            if re.search(pattern, text_lower, flags=re.IGNORECASE):
                if skill not in found_skills:
                    found_skills.append(skill)

    return found_skills


def call_groq(prompt, task_name):
    """Call Groq and return the generated Markdown text."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful career advisor. Return clean, readable Markdown. "
                        "Use meaningful headings, blank lines, bold labels, and bullet or numbered "
                        "lists where appropriate. Use Markdown tables only when they improve clarity. "
                        "Do not return HTML, code fences around the whole answer, or unrelated commentary."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        return content.strip() if content else f"Unable to generate {task_name}. Please try again."
    except Exception:
        logging.exception("%s generation failed.", task_name)
        return (
            f"**Unable to generate {task_name}.** Please check your Groq API key, "
            "model name, and network connection, then try again."
        )


def generate_cover_letter(resume_text, job_role):
    prompt = f"""
Write a professional cover letter for the role of **{job_role}**, based only on the candidate's resume.
Do not invent qualifications, achievements, or experience that are not present in the resume.

Formatting requirements:
- Use a professional letter format.
- Use short, readable paragraphs.
- Do not include Markdown tables.
- Do not wrap the response in a code block.

Resume:
{resume_text}
"""
    return call_groq(prompt, "cover letter")


def generate_interview_qa(resume_text, job_role):
    prompt = f"""
Generate 5 relevant interview questions and suggested answers for the role of **{job_role}**,
based on the candidate's resume. Do not invent experience that is not mentioned.

Formatting requirements:
- Use a heading: "Interview Questions and Answers".
- Number each question.
- Make each question bold.
- Put each answer under its question and label it **Answer:**.
- Leave a blank line between question-and-answer pairs.
- Do not wrap the response in a code block.

Resume:
{resume_text}
"""
    return call_groq(prompt, "interview Q&A")


def suggest_career_paths(resume_text):
    prompt = f"""
Act as a professional career advisor. Suggest 3 suitable career paths based on the candidate's resume.

For each career path, include:
- A clear Markdown heading with the career path name.
- A **Why it fits** subsection with 2-3 bullet points grounded in the resume.
- A **Skills to sharpen** subsection with a numbered list of practical skills or technologies to learn.

After the three paths, include a **Common Skill-Gap Checklist** using a Markdown table with columns
"Area" and "Skills to Learn". Finish with a short **Recommended Next Steps** section.

Use clear headings, concise paragraphs, bullet points, numbered lists, and valid Markdown tables.
Avoid unsupported claims about the candidate. Do not wrap the response in a code block.

Resume:
{resume_text}
"""
    return call_groq(prompt, "career suggestions")


@app.route("/", methods=["GET", "POST"])
def index():
    job_role = None
    cover_letter = None
    error = None
    warning = None
    skill_gaps = []
    resume_score = None
    extracted_skills = []
    interview_qa = None
    career_suggestions = None
    all_roles = list(required_skills_by_role.keys())

    if request.method == "POST":
        if "resume" in request.files and request.files["resume"].filename:
            try:
                uploaded_file = request.files["resume"]

                if uploaded_file.filename.lower().endswith(".pdf"):
                    resume_text = extract_text_from_pdf(uploaded_file)

                    if not resume_text:
                        session.pop("resume_text", None)
                        error = (
                            "No readable text was found in the PDF. "
                            "Please upload a text-based PDF."
                        )
                    else:
                        session["resume_text"] = resume_text
                        resume_vector = vectorizer.transform([resume_text.lower()])
                        job_role = classifier.predict(resume_vector)[0]
                else:
                    error = "Please upload a valid PDF file."
            except Exception:
                logging.exception("Resume upload or classification failed.")
                error = (
                    "An error occurred while processing the resume. "
                    "Please check the PDF and try again."
                )

        elif "manual_role" in request.form:
            selected_role = request.form.get("manual_role", "").strip()
            if selected_role:
                job_role = selected_role

        resume_text = session.get("resume_text")

        if resume_text and job_role:
            standardized_role = role_mapping.get(job_role.lower(), job_role)
            extracted_skills = extract_skills(resume_text)

            required_skills = normalized_skill_map.get(standardized_role.lower())

            if required_skills:
                extracted_lower = {skill.lower() for skill in extracted_skills}
                skill_gaps = [
                    skill for skill in required_skills
                    if skill.lower() not in extracted_lower
                ]

                resume_score = int(
                    ((len(required_skills) - len(skill_gaps)) / len(required_skills)) * 100
                )

                if resume_score >= 30:
                    cover_letter = generate_cover_letter(resume_text, standardized_role)
                    interview_qa = generate_interview_qa(resume_text, standardized_role)

                career_suggestions = suggest_career_paths(resume_text)
            else:
                warning = (
                    "Skill gap analysis is not available for "
                    f"the role: {job_role}."
                )
        elif not resume_text and not error:
            warning = "Please upload a resume before proceeding."

    return render_template(
        "index.html",
        job_role=job_role,
        cover_letter=cover_letter,
        error=error,
        warning=warning,
        resume_score=resume_score,
        skill_gap=skill_gaps,
        extracted_skills=extracted_skills,
        all_roles=all_roles,
        interview_qa=interview_qa,
        career_suggestions=career_suggestions,
    )


@app.route("/download_pdf")
def download_pdf():
    cover_letter = request.args.get("cover_letter", "").strip()

    if not cover_letter:
        return "No cover letter to download.", 400

    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        for line in cover_letter.splitlines():
            pdf.multi_cell(0, 8, line.encode("latin-1", "replace").decode("latin-1"))

        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        pdf_output = BytesIO(pdf_bytes)
        pdf_output.seek(0)

        return send_file(
            pdf_output,
            as_attachment=True,
            download_name="cover_letter.pdf",
            mimetype="application/pdf",
        )
    except Exception:
        logging.exception("PDF generation failed.")
        return "Could not generate the PDF. Please check the cover letter text.", 500


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
