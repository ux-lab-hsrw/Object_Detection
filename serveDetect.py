
from flask import Flask, request

import detectImage

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello World!"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'img' not in request.files:
        return "not found"
    else:
        file = request.files['img']
        file.save(file.filename)
        exec(open('detectImage.py').read())
        #print(file.filename + " contains " + answer)
        return "File successfully uploaded!:" + file.filename

if __name__ == '__main__':
    app.run(host='192.168.0.107', port=8080)