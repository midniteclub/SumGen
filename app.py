from flask import Flask, request, render_template, redirect, url_for, session
import fitz  # PyMuPDF
import os
from werkzeug.utils import secure_filename
from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
import docx
from diffusers import DiffusionPipeline
import torch
from io import BytesIO
import base64

app = Flask(__name__)
app.secret_key = '50f179339739b099b41dd8427a4b7803'

# initializing tuned model
tuned_model_path = "./tuned_model"
tuned_model = BartForConditionalGeneration.from_pretrained(tuned_model_path)
tuned_tokenizer = BartTokenizer.from_pretrained(tuned_model_path)

# initializing Pegasus model
pegasus_tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
pegasus_model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")

# initializing stable diffusion
diffusion_pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# pymupdf (fitz)
def extract_text_from_pdf(file_stream):
    try:
        doc = fitz.open(stream=file_stream.read(), filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text, None  # return text and None for error
    except Exception as e:
        return None, str(e)  # return None for text and the error message

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
def generate_summary(text, summary_model='xsum', max_length=150, min_length=40):
    if summary_model == 'xsum':
        tokenizer = tuned_tokenizer
        model = tuned_model
    elif summary_model == 'pegasus':
        tokenizer = pegasus_tokenizer
        model = pegasus_model
    else:
        return "Invalid summary model selected."

    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    summary_ids = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                 max_length=max_length, min_length=min_length,
                                 length_penalty=2.0, num_beams=4, early_stopping=True)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

# stable diffusion 
def generate_image_from_text(text):
    image = diffusion_pipeline(text).images[0]
    
    buf = BytesIO()
    image.save(buf, format='PNG')
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return f'data:image/png;base64,{image_base64}'


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    error_message = None
    image_data = None  # Initialize outside the conditional logic
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        summary_model = request.form.get('summary_model', 'xsum')
        generate_image = request.form.get('generate_image', 'no')

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
            elif not error:
                summary = generate_summary(text, summary_model)
                if generate_image == 'yes':
                    image_data = generate_image_from_text(summary)
            else:
                error_message = f"Failed to extract text: {error}"
        else:
            error_message = "Unsupported file type."
                
        return render_template('summary2.html', summary=summary, image_data=image_data, error=error_message)

    return render_template('upload2.html', error=error_message)


if __name__ == '__main__':
    app.run(debug=True)