from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import PyPDF2
import sys
import paramiko
import time

app = Flask(__name__, static_url_path='/static', static_folder='static')#we are calling our application "app"
app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = 'uploads' #where the input files will be uploaded
app.config['ALLOWED_EXTENSIONS'] = {'pdf'} #only pdf extension is accepted for the input files

def allowed_file(filename): #checking if the uploaded file has the right extenstion
    print(f"Checking if file allowed: {filename}")
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(pdf_path): #saving the content of the filesas a string
    try:
        print(f"Extracting text from PDF: {pdf_path}")
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() if page.extract_text() else ""
        print(text)
        return text
    except Exception as e:
        sys.stderr.write(f'PDF Extraction Error: {str(e)}\n')
        return None
def execute_ssh_command(host, port, username, private_key_path, password, extracted_text):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    private_key = paramiko.Ed25519Key(filename=private_key_path, password=password)
    ssh.connect(host, port=port, username=username, pkey=private_key)
    print("Connected to SSH server, executing commands...")

    commands = [
        "srun --nodes=1 --nodelist=lenurple --pty /bin/bash -l",
        "cd /home/u590531/llm/",
        "conda activate hf",
        f"echo '{extracted_text}' | python 7bb.py"
    ]

    output_ready = False
    start_time = time.time()
    max_wait_time = 300  # 5 minutes in seconds

    for cmd in commands:
        stdin, stdout, stderr = ssh.exec_command(cmd)
        if '.py' in cmd:
            print("Waiting for 7bb.py to complete...")
            while not output_ready and (time.time() - start_time) < max_wait_time:
                line = stdout.readline()
                if line.strip() == "OUTPUT_READY":  # Signal from 7bb.py
                    output_ready = True
                    break
                elif line.strip() == "":  # EOF without signal
                    break
                else:
                    time.sleep(2)  # Wait before checking again

            if output_ready:
                output = stdout.read().decode()  # Only decode this part
                print("Received output from 7bb.py")
                ssh.close()
                return output

    ssh.close()
    if output_ready:
        return "No output received from 7b.py"
    else:
        return "Timeout or Error in 7bb.py execution"


@app.route('/')
def index():
    print("Serving index page")
    return render_template('interface.html')

@app.route('/upload', methods=['POST']) #this function uploads the pdf files into the folder
def upload_file():
    print("Received file upload request")
    if 'file' not in request.files:
        print("No file part in the request")
        return jsonify({'success': False, 'message': 'No file part'})

    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Saving uploaded file: {pdf_path}")
        file.save(pdf_path)

        extracted_text = extract_text_from_pdf(pdf_path)
        if extracted_text is None:
            print("Failed to extract text from PDF")
            return jsonify({'success': False, 'message': 'Failed to extract text from PDF'})

        host = 'aurometalsaurus.uvt.nl'
        port = 22
        username = 'u590531'
        private_key_path = "C:\\Users\\ooo\\Desktop\\keys\\zan.pem"
        password = 'tesla'

        output = execute_ssh_command(host, port, username, private_key_path, password, extracted_text)
        # Instead of returning a JSON response, render the summary template with the output
        return render_template('summary.html', output=output)

    print("Invalid file type or no file selected")
    return jsonify({'success': False, 'message': 'Invalid file type or no file selected'})

if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(debug=True, port=80)
