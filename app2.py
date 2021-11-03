from flask import Flask, render_template, request
import os
from PIL import Image
import cv2
import numpy as np
from load_img import load_image_into_numpy_array
from run_inference import run_inference_for_single_image
from graph_load import detection_graph, category_index
from matplotlib import pyplot as plt

IMAGE_SIZE = (12, 8)
from object_detection.utils import visualization_utils as vis_util

from werkzeug.utils import secure_filename


UPLOAD_FOLDER = './uploads'
PEOPLE_FOLDER = os.path.join('static', 'saveimg')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER2'] = PEOPLE_FOLDER

@app.route("/")
def home():
    return render_template("home2.html")


@app.route("/function", methods=["POST", 'GET'])
def upload():
    if request.method == "POST":
        file = request.files['files']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
        img = Image.open(file)
        image_np = load_image_into_numpy_array(img)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
        plt.imsave('./static/saveimg/image1.jpg', image_np)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER2'], 'image1.jpg')
        # return render_template("index.html", user_image=full_filename)
        return render_template("home3.html", user_image=full_filename)
    # return render_template("home3.html")



if __name__ == "__main__":
    app.run(debug=True)
