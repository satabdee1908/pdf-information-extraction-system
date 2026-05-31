from groq import Groq
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
import pdfplumber
import os
import re
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import joblib
import spacy
import json
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Users\satab\Downloads\Release-26.02.0-0\poppler-26.02.0\Library\bin"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

model = joblib.load("document_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

nlp = spacy.load("en_core_web_sm")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.get("/")
def home():
    return {"message": "PDF and Image OCR Parser API with ML and NLP is working"}


def clean_text(text):
    text = text.replace("\n\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def detect_document_type(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]


def extract_entities(text):
    doc = nlp(text)

    entities = {
        "persons": [],
        "organizations": [],
        "dates": [],
        "locations": []
    }

    bad_person_words = [
        "academic", "transcript", "order", "dean", "resume",
        "identity", "card", "subject", "notice", "problem solving",
        "education", "skills"
    ]

    bad_org_words = ["pdf", "email", "phone"]

    for ent in doc.ents:
        value = ent.text.strip()
        value_lower = value.lower()

        if len(value) < 3:
            continue

        if ent.label_ == "PERSON":
            if any(word in value_lower for word in bad_person_words):
                continue
            if any(char.isdigit() for char in value):
                continue
            if "www" in value_lower or "linkedin" in value_lower:
                continue
            entities["persons"].append(value)

        elif ent.label_ == "ORG":
            if any(word in value_lower for word in bad_org_words):
                continue
            if re.search(r"\d{10}", value):
                continue
            entities["organizations"].append(value)

        elif ent.label_ == "DATE":
            if re.search(r"\d{10}", value):
                continue
            entities["dates"].append(value)

        elif ent.label_ in ["GPE", "LOC"]:
            if any(char.isdigit() for char in value):
                continue
            entities["locations"].append(value)

    regex_dates = re.findall(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        text
    )

    written_dates = re.findall(
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\b",
        text
    )

    entities["dates"].extend(regex_dates)
    entities["dates"].extend(written_dates)

    for key in entities:
        entities[key] = list(set(entities[key]))

    return entities


def extract_text_with_ocr_from_pdf(file_path):
    ocr_text = ""

    images = convert_from_path(
        file_path,
        dpi=300,
        poppler_path=POPPLER_PATH
    )

    for image in images:
        text = pytesseract.image_to_string(image)
        ocr_text += text + "\n"

    return ocr_text


def extract_text_from_image(file_path):
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    return text


def extract_headings(text):
    headings = []

    heading_keywords = [
        "NOTICE", "SUBJECT", "SUMMARY", "EDUCATION", "SKILLS",
        "EXPERIENCE", "PROJECTS", "CERTIFICATIONS",
        "IMPORTANT INSTRUCTIONS", "APPLICATIONS", "CHALLENGES",
        "CONCLUSION", "INTRODUCTION"
    ]

    for line in text.split("\n"):
        line = line.strip()

        if len(line) < 3:
            continue

        if line.isupper() and line not in headings:
            headings.append(line)

        for keyword in heading_keywords:
            if line.upper().startswith(keyword) and line not in headings:
                headings.append(line)

    return headings


def parse_resume(text):
    details = {}

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    details["name"] = lines[0] if lines else None

    email = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    details["email"] = email.group() if email else None

    phone = re.search(r"(\+91[\s-]?)?\d{10}", text)
    details["phone"] = phone.group() if phone else None

    linkedin = re.search(r"(https?://)?(www\.)?linkedin\.com/[^\s|]+", text)
    details["linkedin"] = linkedin.group() if linkedin else None

    skill_keywords = [
        "Python", "Java", "C", "C++", "HTML", "CSS", "JavaScript",
        "SQL", "DBMS", "Machine Learning", "Artificial Intelligence",
        "Data Structures", "FastAPI", "Git", "GitHub"
    ]

    skills = []

    for skill in skill_keywords:
        if re.search(r"\b" + re.escape(skill) + r"\b", text, re.IGNORECASE):
            skills.append(skill)

    details["skills"] = skills

    education = []
    education_keywords = [
        "B.Tech", "Bachelor", "CGPA", "Class X", "Class XII",
        "CBSE", "ICSE", "Engineering", "University", "School"
    ]

    for line in lines:
        if any(keyword.lower() in line.lower() for keyword in education_keywords):
            education.append(line)

    details["education"] = education

    return details


def parse_invoice(text):
    details = {}

    invoice_number = re.search(
        r"(invoice\s*(number|no|#)?[:\s-]*)([A-Za-z0-9/-]+)",
        text,
        re.IGNORECASE
    )
    details["invoice_number"] = invoice_number.group(3) if invoice_number else None

    date = re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)
    details["date"] = date.group() if date else None

    amount = re.search(r"(₹|Rs\.?|INR)\s?\d+[,\d]*", text)
    details["amount"] = amount.group() if amount else None

    email = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    details["email"] = email.group() if email else None

    phone = re.search(r"(\+91[\s-]?)?\d{10}", text)
    details["phone"] = phone.group() if phone else None

    return details


def parse_general_document(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    title = lines[0] if lines else None
    headings = extract_headings(text)

    content_lines = []

    for line in lines:
        clean_line = line.replace("•", "").strip()

        if clean_line.startswith("e "):
            clean_line = clean_line[2:].strip()

        if clean_line == title:
            continue

        if clean_line in headings:
            continue

        if len(clean_line.split()) < 4:
            continue

        if clean_line.lower().startswith(("email:", "phone:", "date:")):
            continue

        content_lines.append(clean_line)

    key_points = content_lines[:5]
    summary = " ".join(content_lines[:3])

    words = re.findall(r"\b[a-zA-Z]{5,}\b", text.lower())

    stopwords = [
        "there", "their", "about", "which", "would", "could",
        "should", "these", "those", "from", "with", "have",
        "this", "that", "were", "been", "will", "into",
        "also", "your", "they", "them", "using", "such"
    ]

    filtered_words = [word for word in words if word not in stopwords]

    word_count = {}

    for word in filtered_words:
        word_count[word] = word_count.get(word, 0) + 1

    important_keywords = sorted(
        word_count,
        key=word_count.get,
        reverse=True
    )[:10]

    return {
        "title": title,
        "key_points": key_points,
        "summary": summary,
        "important_keywords": important_keywords
    }
def generate_ai_summary(text):
    try:
        short_text = text[:4000]

        prompt = f"""
You are an AI document intelligence assistant.

Analyze the following extracted document text.

The document may be a resume, invoice, notice, article, report, marksheet, letter, form, or general document.

Return ONLY valid JSON.
Do not use markdown.
Do not use ```json.
Do not add explanation.

JSON format:
{{
    "suggested_document_type": "",
    "summary": "",
    "key_points": ["", "", "", "", ""],
    "important_entities": {{
        "people": [],
        "organizations": [],
        "dates": [],
        "locations": []
    }},
    "important_fields": {{}}
}}

Rules:
- Identify the document type from the content.
- Generate a professional summary.
- Extract the 5 most important points.
- Extract important people, organizations, dates and locations.
- Ignore OCR noise and meaningless text.
- Keep output concise and accurate.

Document text:
{short_text}
"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        ai_text = response.choices[0].message.content.strip()
        ai_text = ai_text.replace("```json", "").replace("```", "").strip()

        return json.loads(ai_text)

    except Exception as e:
        return {
            "error": f"Groq AI error: {str(e)}"
        }
@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    extracted_text = ""
    total_pages = 0
    extraction_method = ""

    file_extension = file.filename.lower().split(".")[-1]

    if file_extension == "pdf":
        extraction_method = "Normal PDF Text Extraction"

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)

            for page in pdf.pages:
                text = page.extract_text()

                if text:
                    extracted_text += text + "\n"

        if len(extracted_text.strip()) < 50:
            extracted_text = extract_text_with_ocr_from_pdf(file_path)
            extraction_method = "OCR from Scanned PDF"

    elif file_extension in ["jpg", "jpeg", "png"]:
        total_pages = 1
        extracted_text = extract_text_from_image(file_path)
        extraction_method = "OCR from Image"

    else:
        return {
            "error": "Unsupported file type. Please upload PDF, JPG, JPEG, or PNG."
        }

    cleaned_text = clean_text(extracted_text)

    document_type = detect_document_type(cleaned_text)

    entities = extract_entities(cleaned_text)

    headings = extract_headings(cleaned_text)

    if document_type == "resume":
        parsed_data = parse_resume(cleaned_text)

    elif document_type == "invoice":
        parsed_data = parse_invoice(cleaned_text)
    
    else:
        parsed_data = parse_general_document(cleaned_text)
    ai_output = generate_ai_summary(cleaned_text)
    if document_type == "resume":
        return {
            "filename": file.filename,
            "pages": total_pages,
            "document_type": document_type,
            "classification_method": "ML model using TF-IDF + Logistic Regression",
            "extraction_method": extraction_method,
            "resume_data": parsed_data,
            "ai_method": "Groq Llama 3.3 70B",
            "ai_output": ai_output
        }

    return {
        "filename": file.filename,
        "pages": total_pages,
        "document_type": document_type,
        "classification_method": "ML model using TF-IDF + Logistic Regression",
        "nlp_method": "spaCy en_core_web_sm NER with filtering",
        "ai_method": "Groq Llama 3.3 70B",
        "extraction_method": extraction_method,
        "entities": entities,
        "headings": headings,
        "parsed_data": parsed_data,
        "preview_text": cleaned_text[:1000],
        "ai_output": ai_output
    }