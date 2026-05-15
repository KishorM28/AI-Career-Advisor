from dotenv import load_dotenv
import os
from flask import Flask, render_template, request, session, send_file
import fitz  # PyMuPDF
import pickle
from groq import Groq
import re
import logging
from io import BytesIO
from fpdf import FPDF

app = Flask(__name__)
app.secret_key = "your-super-secret-key"
logging.basicConfig(level=logging.INFO)

# Load model and vectorizer
with open('resume_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

load_dotenv()

# Groq setup
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

required_skills_by_role = {
    'AI Engineer': ['Python', 'TensorFlow', 'PyTorch', 'Deep Learning', 'ML'],
    'Data Scientist': ['Python', 'Machine Learning', 'Statistics', 'Pandas', 'SQL', 'Scikit-learn', 'TensorFlow'],
    'Web Developer': ['HTML', 'CSS', 'JavaScript', 'React', 'Node.js'],
    'Designer': ['Figma', 'Adobe XD', 'UI/UX', 'Sketch', 'Visual Design'],
    'Full Stack Developer': ['HTML', 'CSS', 'JavaScript', 'React', 'Node.js', 'Express', 'MongoDB'],
    'Backend Developer': ['Node.js', 'Express', 'MongoDB', 'SQL', 'APIs'],
    'Frontend Developer': ['HTML', 'CSS', 'JavaScript', 'React', 'UI/UX'],
    'DevOps Engineer': ['Docker', 'Kubernetes', 'CI/CD', 'AWS', 'Terraform'],
    'Mobile App Developer': ['Flutter', 'React Native', 'Java', 'Kotlin', 'Swift'],
    'Cloud Engineer': ['AWS', 'Azure', 'GCP', 'Terraform', 'Linux'],
    'ML Engineer': ['Python', 'Scikit-learn', 'Deep Learning', 'TensorFlow', 'MLOps'],
    'Database Admin': ['SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Database Design'],
    'Graphic Designer': ['Adobe Photoshop', 'Illustrator', 'InDesign', 'Sketch', 'Figma', 'Typography', 'Branding', 'UI/UX', 'Video Editing'],
    'Musician': ['Music Theory', 'Composition', 'Sibelius', 'Ableton', 'Instrumental Skills', 'Recording', 'Sound Editing'],
    'Dancer': ['Choreography', 'Stage Performance', 'Classical Dance', 'Hip-hop', 'Contemporary', 'Flexibility', 'Rhythm'],
    'Actor': ['Stage Presence', 'Improvisation', 'Dialogue Delivery', 'Script Reading', 'Voice Modulation', 'Screen Acting'],
    'Teacher': ['Classroom Management', 'Lesson Planning', 'Curriculum Design', 'Educational Tools', 'Assessment', 'Communication'],
    'Business Analyst': ['Business Analysis', 'Stakeholder Communication', 'Requirement Gathering', 'SQL', 'Excel', 'UML'],
    'Digital Marketer': ['SEO', 'Google Analytics', 'Social Media Marketing', 'Email Marketing', 'AdWords', 'Content Strategy'],
    'Mechanical Engineer': ['CAD', 'SolidWorks', 'Thermodynamics', 'MATLAB', 'Mechanical Design'],
    'Civil Engineer': ['AutoCAD', 'Structural Analysis', 'Construction Management', 'Site Supervision'],
    'Doctor': ['Diagnosis', 'Patient Care', 'Medical Knowledge', 'Clinical Procedures', 'EMR Systems'],
    'Nurse': ['Patient Monitoring', 'Medication Administration', 'Emergency Care', 'Medical Records', 'Compassionate Care']
}

normalized_skill_map = {k.lower(): v for k, v in required_skills_by_role.items()}
role_mapping = {k.lower(): k for k in required_skills_by_role}
role_mapping.update({
    'data science': 'Data Scientist',
    'ml engineer': 'ML Engineer',
    'ai engineer': 'AI Engineer',
    'fullstack developer': 'Full Stack Developer',
    'full stack developer': 'Full Stack Developer',
    'frontend developer': 'Frontend Developer',
    'backend developer': 'Backend Developer',
    'web developer': 'Web Developer',
    'devops engineer': 'DevOps Engineer',
    'cloud engineer': 'Cloud Engineer',
    'business analyst': 'Business Analyst',
    'digital marketer': 'Digital Marketer',
    'graphic designer': 'Graphic Designer',
    'mobile developer': 'Mobile App Developer',
    'teacher': 'Teacher',
    'actor': 'Actor',
    'dancer': 'Dancer',
    'doctor': 'Doctor',
    'nurse': 'Nurse',
    'designer': 'Designer',
    'musician': 'Musician',
    'mechanical engineer': 'Mechanical Engineer',
    'civil engineer': 'Civil Engineer'
})

def extract_text_from_pdf(file_storage):
    text = ""
    doc = fitz.open(stream=file_storage.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

def extract_skills(text):
    known_skills = set(skill.lower() for skills in required_skills_by_role.values() for skill in skills)
    resume_words = set(re.findall(r'\b\w+\b', text.lower()))
    return list(resume_words.intersection(known_skills))

def generate_cover_letter(resume_text, job_role):

    prompt = f"""
    Write a professional cover letter
    for the role of {job_role}
    based on this resume.

    Resume:
    {resume_text}
    """

    try:

        response = client.chat.completions.create(

            model="llama-3.1-8b-instant",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"Error generating cover letter: {e}"
def generate_interview_qa(resume_text, job_role):

    prompt = f"""
    Generate 5 interview questions
    and answers for {job_role}

    Resume:
    {resume_text}
    """

    try:

        response = client.chat.completions.create(

            model="llama-3.1-8b-instant",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"Error generating interview Q&A: {e}"

def suggest_career_paths(resume_text):

    prompt = f"""
    Suggest 3 career paths
    based on this resume.

    Resume:
    {resume_text}
    """

    try:

        response = client.chat.completions.create(

            model="llama-3.1-8b-instant",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"Error generating career suggestions: {e}"

@app.route('/', methods=['GET', 'POST'])
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

    if request.method == 'POST':
        if 'resume' in request.files:
            try:
                file = request.files['resume']
                if file and file.filename.endswith('.pdf'):
                    resume_text = extract_text_from_pdf(file).strip()
                    session['resume_text'] = resume_text
                    resume_vector = vectorizer.transform([resume_text.lower()])
                    job_role_raw = classifier.predict(resume_vector)[0]
                    job_role = job_role_raw
                else:
                    error = "Please upload a valid PDF file."
            except Exception as e:
                logging.exception("Resume upload error:")
                error = f"An error occurred: {e}"
        elif 'manual_role' in request.form:
            job_role = request.form['manual_role']

        resume_text = session.get('resume_text')
        if resume_text and job_role:
            standardized_role = role_mapping.get(job_role.lower(), job_role)
            extracted_skills = extract_skills(resume_text)
            required_skills = normalized_skill_map.get(standardized_role.lower())

            if required_skills:
                skill_gaps = [s for s in required_skills if s.lower() not in map(str.lower, extracted_skills)]
                resume_score = int((len(required_skills) - len(skill_gaps)) / len(required_skills) * 100)

                if resume_score >= 30:
                    cover_letter = generate_cover_letter(resume_text, standardized_role)
                    interview_qa = generate_interview_qa(resume_text, standardized_role)
                    career_suggestions = suggest_career_paths(resume_text)
                else:
                    career_suggestions = suggest_career_paths(resume_text)
            else:
                warning = f"⚠️ Skill gap analysis not available for the role: {job_role}."
        elif not resume_text:
            warning = "⚠️ Please upload a resume before proceeding."

    return render_template("index.html",
                           job_role=job_role,
                           cover_letter=cover_letter,
                           error=error,
                           warning=warning,
                           resume_score=resume_score,
                           skill_gap=skill_gaps,
                           extracted_skills=extracted_skills,
                           all_roles=all_roles,
                           interview_qa=interview_qa,
                           career_suggestions=career_suggestions)

@app.route('/download_pdf')
def download_pdf():
    cover_letter = request.args.get('cover_letter', '')
    if not cover_letter:
        return "No cover letter to download.", 400

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in cover_letter.split('\n'):
        pdf.multi_cell(0, 10, line)

    # Output PDF as string and encode to bytes
    pdf_bytes = pdf.output(dest='S').encode('latin-1')

    # Write bytes to BytesIO stream
    pdf_output = BytesIO()
    pdf_output.write(pdf_bytes)
    pdf_output.seek(0)

    return send_file(pdf_output, as_attachment=True, download_name="cover_letter.pdf", mimetype='application/pdf')


if __name__ == '__main__':
    app.run(debug=True)