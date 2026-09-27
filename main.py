import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
env_path = Path(__file__).resolve().parent /".env"
load_dotenv(dotenv_path=env_path)
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

import re
import io 
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
import pdfplumber
from google import genai
from google.genai import types
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

app = FastAPI(title="BIAS REMOVAL AND ANONYMIZATION API LAYER")

client = genai.Client()

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extracts raw text safely from uploaded PDF bytes."""
    extracted_text = ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
    except Exception as e:
       raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")

    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="Uploaded PDF contains no extractable text.")

    return extracted_text

def code_layer_processing(text: str) -> str:
    """
    code preprocessing Layer: Mask direct identifieres via Regex
    before sending to Gemini to ensure Zero-data leakage.
    """
    text = re.sub(r'[\w\.-]+@[\w\.-]+\. \w+','[CANDIDATE_EMAIL]', text)
    text = re.sub(r'(\+?\d{1,3}[-. \s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-. \s]?\d{4}', '[CANDIDATE_PHONE]', text)
    text = re.sub(r'https?://github\.com/[\w-]+', 'https://github.com/[ANONYMIZER_USER]', text)
    text = re.sub(r'https?://(www\.)?linkedin\.com/in/[\w-]+', 'https://linkedin.com/in/[ANONYMIZER_USER]', text)

    return text

def anonymize_with_gemini(preprocessed_text: str) -> str:
    """
    Gemini Layer: Identifies direct/indirect bias, removes unwanted info,
    anonymizes college names, CGPA/Grades and return a structured profile.
    """
    system_instruction = """
    You are an expert HR AI trained to eliminate hiring bias.
    your task is to take the provided resume text and create a completely anonymized version

    STRICT ANONYMIZATION RULES:
    1. Remove Direct Identifiers: Name, Gender, Address, DOB, Age, Marital ststus, Nationality
    2. Remove / Anonymize Indirect Bias Indicators:
       - School/college/University Names: Replace with "Recognized University / Institution".
       - Degree Bias Removal (B.Tech / B.E / B.Sc / Diplomo / BCA / BA / Tecnologo / BEng / VB): Standardize all undergraduate degree names to a neutral format such as "Bachelor's Degree in [Field/Major]" (e.g., replace "B.Tech in Computer Science" with "Bachelor's Degree in Computer Science) to avoid degree -type and institution tier preference bias. Preserve the major/field of study.
       - Graduation years or explicit timelines that imply age : omit the years 
       - Academic scores / CGPA / Percentages: Anonymize or omit exact CGPA/Percentage to prevent academic tier/grade bias (e.g., replace with "[DEGREE_COMPLETED]").
       - Gender-specific pronouns: Use gender-neutral terms or third-person phrasing.
    3. GITHUB & PORTFOLIO ANONYMIZATION:
       - Do NOT remove GitHub or portfolio project references completely.
       - Anonymize personal identifiers in repository URLs or username handles (e.g.,'github.com/[ANONYMIZED USER]').
       - Retain all technical project description, code contributions, open-source work, and repository titles so skills are fully visible.
    4. KEEP AND HIGHLIGHT:
       - Technical and Soft Skills.
       - Work Experience, Roles, and Scope of Responsibilities.
       - Measurable Achivements, Projects, and Metrics (business outcomes).
       - Certifications (Remove institution name if it reveals location/tier bias).
    FORMATTING INSTRUCTIONS:
       - Do NOT use Markdown formatting like double asterisks (**) or single asterisks (*) anywhere in the response text. Return pure clean text
       - Output MUST be clean, peofessional Markdown.
       - Start directly with the title: '# ANONYMIZED CANDIDATE PROFILE'.
       - Use sections: ## Professional Summary, ## Technical & Core Skills, ## Work Experience, ## Key project, ## Education & Certifications.
       - Do NOT include any intro, meta-comments, or polite conversational text before/after.
       """

    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=preprocessed_text,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
            ) ,
        )
        return response.text
    except Exception as e:
        raise HTTPException (status_code=500, detail=f"Gemini API Processing Error: {str(e)}")

def convert_markdown_to_pdf(markdown_text: str) -> io.BytesIO:
    """Generates a clean PDF document from Markdown text using Reportlab."""
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)

    styles = getSampleStyleSheet()
    normal_style =styles['Normal']
    normal_style.fontsize = 10
    normal_style.leading = 14

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontsize=16,
        leading=20,
        spaceAfter=12
    )

    heading_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        spaceBefore=10,
        spaceAfter=6
    )

    story =[]
    lines = markdown_text.split('\n')

    for line in lines:
        line_str = line.strip()
        if not line_str:
            story.append(Spacer(1, 4))
            continue

        if line_str.startswith('# '):
            text = line_str.replace('# ','')
            story.append(Paragraph(f"<b>{text}</b>", title_style))
        elif line_str.startswith('## '):
            text = line_str.replace('## ','')
            story.append(Paragraph(f"<b>{text}</b>", heading_style))
        elif line_str.startswith('- ') or line_str.startswith('* '):
            text = line_str[2:]
            text = text.replace('&', '&amp;').replace('<','&lt;').replace('>','&gt;')
            story.append(Paragraph(f"&bull;{text}", normal_style))
        else:
            text = line_str.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
            story.append(Paragraph(text, normal_style))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer
    
@app.post("/anonymize-resume/")
async def anonymize_resume(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    raw_text = extract_text_from_pdf(pdf_bytes)
    preprocessed_text = code_layer_processing(raw_text)
    anonymized_markdown = anonymize_with_gemini(preprocessed_text)
    output_pdf_buffer = convert_markdown_to_pdf(anonymized_markdown)

    return StreamingResponse(
        output_pdf_buffer,
        media_type="application/pdf",
        headers={"content-Disposition": "attachment; filename=Anonymized_Resume.pdf"}
    ) 

if __name__ =="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    