# utils.py
import spacy

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

# Define known skills
KNOWN_SKILLS = {
    "Python", "Machine Learning", "Pandas", "Numpy", "Statistics", "SQL",
    "HTML", "CSS", "JavaScript", "React", "Node.js",
    "TensorFlow", "PyTorch", "Deep Learning", "NLP",
    "Java", "OOP", "Spring Boot", "REST API",
    "Docker", "Kubernetes", "CI/CD", "Linux", "AWS"
}

def extract_resume_skills(text):
    """
    Extracts relevant skills from resume text by matching against a predefined list of known skills.

    Args:
        text (str): The input resume text.

    Returns:
        set: A set of matched skills found in the resume.
    """
    lower_known_skills = {skill.lower() for skill in KNOWN_SKILLS}
    doc = nlp(text.lower())

    tokens = set()

    # Capture noun chunks and important tokens
    for chunk in doc.noun_chunks:
        tokens.add(chunk.text.strip())
    for token in doc:
        if not token.is_stop and not token.is_punct:
            tokens.add(token.text.strip())

    matched_skills = {
        skill for skill in KNOWN_SKILLS
        if skill.lower() in tokens or any(skill.lower() in t for t in tokens)
    }

    return matched_skills
