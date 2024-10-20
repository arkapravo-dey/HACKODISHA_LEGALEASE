from flask import Flask, render_template, request, send_from_directory
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer
from gtts import gTTS
import PyPDF2
import os

app = Flask(__name__)

# Load Pegasus model and tokenizer for summarization
pegasus_model_name = "nsi319/legal-pegasus"
pegasus_tokenizer = PegasusTokenizer.from_pretrained(pegasus_model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name)

# Load M2M100 model and tokenizer for translation
m2m100_model_name = "facebook/m2m100_418M"
m2m100_tokenizer = M2M100Tokenizer.from_pretrained(m2m100_model_name)
m2m100_model = M2M100ForConditionalGeneration.from_pretrained(m2m100_model_name)

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_file_path):
    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

# Step 2: Summarize text using Legal Pegasus
def summarize_text(text):
    inputs = pegasus_tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    summary_ids = pegasus_model.generate(
        inputs.input_ids,
        max_length=512,
        num_beams=5,
        length_penalty=2.0,
        repetition_penalty=2.5,
        early_stopping=True
    )
    summary = pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Step 3: Translate text using M2M100
def translate_text(summary, target_lang):
    m2m100_tokenizer.src_lang = "en"
    m2m100_tokenizer.tgt_lang = target_lang
    inputs = m2m100_tokenizer(summary, return_tensors="pt", truncation=True)
    translated_ids = m2m100_model.generate(**inputs, forced_bos_token_id=m2m100_tokenizer.get_lang_id(target_lang))
    translated_summary = m2m100_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    return translated_summary

# Step 4: Convert text to speech using gTTS
def text_to_speech(text, lang, file_name):
    tts = gTTS(text=text, lang=lang, slow=False)
    file_path = os.path.join('static', file_name)
    tts.save(file_path)
    return file_path

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/abstractor', methods=['GET', 'POST'])
def abstractor():
    if request.method == 'POST':
        # Get uploaded PDF
        pdf_file = request.files['pdf_file']
        pdf_path = os.path.join("uploads", pdf_file.filename)
        pdf_file.save(pdf_path)

        # Get selected language
        target_lang = request.form.get('target_lang')

        # Extract, summarize, and translate
        document_text = extract_text_from_pdf(pdf_path)
        english_summary = summarize_text(document_text)
        translated_summary = translate_text(english_summary, target_lang)

        # Convert to speech
        english_audio = text_to_speech(english_summary, lang="en", file_name="english_summary.mp3")
        translated_audio = text_to_speech(translated_summary, lang=target_lang, file_name=f"translated_summary_{target_lang}.mp3")

        return render_template('abstractor.html', 
                               english_summary=english_summary, 
                               translated_summary=translated_summary, 
                               english_audio=english_audio, 
                               translated_audio=translated_audio)
    
    return render_template('abstractor.html')

if __name__ == '__main__':
    # Create necessary directories
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('static'):
        os.makedirs('static')

    app.run(debug=True)

