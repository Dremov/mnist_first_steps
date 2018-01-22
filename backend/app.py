from flask import Flask, request, redirect, jsonify
from flask_cors import CORS
from compute_lrp import ComputeLRP
from PIL import Image
from io import BytesIO
import numpy as np
import base64
import sys,os,io

app = Flask(__name__)
CORS(app)

lrp = ComputeLRP()


@app.route('/lrp')
def compute_lrp():

    return "Success"

@app.route('/image', methods=['POST'])
def recieve_image():
    # check if the post request has the file part
    if 'image' not in request.files:
        print('No image data found!')
        return redirect(request.url)
    
    image_blob = request.files['image']
    image_data_url = image_blob.read()
    image_base64 = str(image_data_url).split(",")[1][:-1]
    image = base64.b64decode(image_base64)
    
    img = Image.open(io.BytesIO(image))
    heatmap, prediction = lrp.compute_lrp(img)
    
    heatmap *= 255

    pil_img = Image.fromarray(heatmap.astype("uint8"))
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    image_base64 = base64.b64encode(buff.getvalue())
    image_base64 = str(image_base64).split("'")[1]

    return jsonify({'success':True, 'prediction': str(prediction), 'heatmap': image_base64})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=81)

