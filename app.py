from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from llama7 import output #here we are importing the variable that contains the summary made by llama

app = Flask(__name__, static_url_path='/static', static_folder='static') #we are calling our application "app"
app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = 'uploads' #where the input files will be uploaded
app.config['ALLOWED_EXTENSIONS'] = {'pdf'} 
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/') 
def index():
    return render_template('interface.html')

@app.route('/upload', methods=['POST'])
def upload_file(): #this function uploads the pdf files into the folder
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})

    files = request.files.getlist('file')

    if not files:
        return jsonify({'success': False, 'message': 'No selected files'})

    file_paths = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_paths.append(file_path)
        else:
            return jsonify({'success': False, 'message': 'Invalid file type! Please upload only pdf.'})

    return jsonify({'success': True, 'message': 'Files uploaded successfully', 'summary': output})

@app.route('/summary')
def summary():
    
    return render_template("summary.html", output=output)

    

if __name__ == '__main__':
    app.run(debug=True, port=80)
