# pytorch 모델을 서빙
import torch
import numpy as np
from torchvision import transforms
from flask import Flask, jsonify, request

from model import CNN
from models.classification.image_classification import testPredictLabel
import os

model = CNN()
model.load_state_dict(torch.load('mnist_model.pt'), strict=False)
model.eval()

normalize = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

app = Flask(__name__)

'''
(GET) image classification model test router
'''
@app.route('/image-classification-test', methods=['GET'])
def imageClassficiatonTest():
    category = testPredictLabel()
    
    # realCategory = predictLabel(img)
    return category


@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    _, result = model.forward(
        normalize(np.array(data['images'], dtype=np.uint8)).unsqueeze(0)).max(1)
    return str(result.item())


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=2431)
