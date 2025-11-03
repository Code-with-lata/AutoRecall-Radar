import os
import json
import pdfplumber
import docx2txt
import tempfile
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import easyocr
from flask import Flask, request, jsonify

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", None) 
if not API_KEY: 
    st.error("Gemini API key not found. Please set GEMINI_API_KEY environment variable.") 
else: 
    genai.configure(api_key=API_KEY) 
# model = genai.GenerativeModel("gemini-2.5-flash")

# OCR Reader for images
ocr_reader = easyocr.Reader(['en'])
app = Flask(__name__)

def extract_text_from_file(file_name):
    upload_file = file_name.filename.lower()
    text = ""
    if upload_file.name.endswith(".txt"):
        text = upload_file.read().decode("utf-8")

    elif upload_file.name.endswith(".pdf"):
        with pdfplumber.open(upload_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

    elif upload_file.name.endswith(".docx"):
        text = docx2txt.process(upload_file)

    return text.strip()    

# Extract text from image (OCR)
def extract_text_from_image(uploaded_image):
    image = Image.open(uploaded_image)
    results = ocr_reader.readtext(image)
    extracted_text = " ".join([res[1] for res in results])
    return extracted_text.strip()

def analyze_report(report_text):
    """
    This function uses Gemini LLM to analyze automotive complaints/service reports
    and outputs a structured JSON containing recall risks, component failures,
    safety score, and preventive actions.
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    
    prompt = f"""
You are an expert Automotive Safety AI. 
Analyze the following automotive complaint, service report, or input data.

⚠ STRICT RULES:
- Output must be ONLY raw JSON. Do not include markdown, backticks, or explanations.
- Focus only on the main risks, failures, and preventive actions. 
- Use clear phrases Only apply in component_failures and preventive_avtions.  
- Safety score must be an integer (1-10).

JSON Output Format:
{{
  "recall_risks": ["Describe high-level recall risks based on the input"],
  "component_failures": ["Short component failures, 2-5 items"],
  "safety_score": integer (1-10, where 10 = highest risk),
  "preventive_actions": ["List actionable steps the manufacturer or safety authorities should take"]
}}

Input:
{report_text}
"""


    response = model.generate_content(prompt)
    return response.text


# ------------------ Cloud Function Endpoint ------------------

@app.route("/", methods=["POST"])
def auto_recall_radar():
    """
    Flask / Cloud Function entry point for analyzing automotive complaints.
    Accepts:
      - text: direct complaint text
      - file: uploaded file (PDF, DOCX, TXT, or Image)
    Returns:
      - JSON structured response from Gemini model
    """
    try:
        report_text = None

        #  Raw JSON input
        if request.is_json:
            data = request.get_json(silent=True)
            if data and "text" in data:
                report_text = data["text"].strip()

        #  Raw plain text in body
        elif request.data and request.data.strip():
            report_text = request.data.decode("utf-8").strip()

        #  Form-data text
        elif "text" in request.form and request.form["text"].strip():
            report_text = request.form["text"].strip()

        #  Check for file input
        elif "file" in request.files:
            file = request.files["file"]
            filename = file.filename.lower()

            if filename.endswith((".png", ".jpg", ".jpeg")):
                report_text = extract_text_from_image(file)
            else:
                report_text = extract_text_from_file(file)

        #  If no valid input
        if not report_text or not report_text.strip():
            return jsonify({"error": "No valid input found. Please provide text or file."}), 400

        #  Analyze with Gemini
        result = analyze_report(report_text)

        #  Parse JSON safely
        try:
            result_json = json.loads(result)
            return jsonify(result_json)
        except json.JSONDecodeError:
            return jsonify({
                "error": "Failed to parse Gemini response as JSON.",
                "raw_output": result
            }), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------ Cloud Function Entry ------------------
def auto_recall_radar_entry(request):
    """Google Cloud Function entry point"""
    with app.app_context():
        return auto_recall_radar()