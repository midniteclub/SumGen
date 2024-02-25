from flask import Flask, request, render_template, redirect, url_for, session
import fitz  # PyMuPDF
import os
from transformers import BartTokenizer, BartForConditionalGeneration
import docx
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = '50f179339739b099b41dd8427a4b7803'

# loading fine-tuned model (trained with optimzed hyperparameters)
model_path = "./tuned_model"
model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained(model_path)

# pymupdf (fitz)
def extract_text_from_pdf(file_stream):
    try:
        doc = fitz.open(stream=file_stream.read(), filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text, None  # Return text and None for error
    except Exception as e:
        return None, str(e)  # Return None for text and the error message

# python-docx
def extract_text_from_docx(file_stream):
    try:
        doc = docx.Document(file_stream)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs]), None
    except Exception as e:
        return None, str(e)

def extract_text_from_txt(file_stream):
    try:
        text = file_stream.read().decode('utf-8')
        return text, None
    except Exception as e:
        return None, str(e)

# decoding tokenizer: generating text
def generate_summary(text, max_length=150, min_length=40):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding="max_length")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    summary_ids = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                 max_length=max_length, min_length=min_length,
                                 length_penalty=2.0, num_beams=4, early_stopping=True)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    error_message = None
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()

        if file_extension in ['pdf', 'docx', 'txt']:
            file.stream.seek(0) 
            if file_extension == 'pdf':
                text, error = extract_text_from_pdf(file.stream)
            elif file_extension == 'docx':
                text, error = extract_text_from_docx(file)
            elif file_extension == 'txt':
                text, error = extract_text_from_txt(file.stream)
            else:
                text, error = None, "Unsupported file type."
            
            if error:
                error_message = f"Failed to extract text: {error}"
            else:
                summary_length = request.form.get('summary_length', 'short')
                if summary_length == 'medium':
                    summary = generate_summary(text, max_length=420, min_length=80) # adjustable max_length and min_length
                elif summary_length == 'long':
                    summary = generate_summary(text, max_length=800, min_length=100) # adjustable max_length and min_length
                else: 
                    summary = generate_summary(text)
                return render_template('summary.html', summary=summary)
        else:
            error_message = "Unsupported file type. Please upload a .pdf, .docx, or .txt file."

    return render_template('upload.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)