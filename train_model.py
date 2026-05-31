from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Sample training data

texts = [

    # Resume
    "skills python java education projects experience linkedin github",
    "resume with technical skills internships projects machine learning",
    "education qualifications programming languages certifications",

    # Invoice
    "invoice number gst total amount payment tax bill",
    "payment invoice amount due tax invoice number",
    "bill receipt gst payment transaction amount",

    # Article
    "abstract introduction conclusion references research paper",
    "article report discussion methodology conclusion",
    "research study analysis journal references summary",

    # General Document
    "semester marks grade subject examination result",
    "student marksheet university semester cgpa",
    "official notice announcement information circular"

]

labels = [

    "resume",
    "resume",
    "resume",

    "invoice",
    "invoice",
    "invoice",

    "article/report",
    "article/report",
    "article/report",

    "general_document",
    "general_document",
    "general_document"
]

# Convert text into numerical vectors

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)

# Train model

model = LogisticRegression()

model.fit(X, labels)

# Save model and vectorizer

joblib.dump(model, "document_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained and saved successfully")