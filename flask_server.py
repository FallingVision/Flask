import torch
import numpy as np
from torchvision import transforms
from flask import Flask, jsonify, request
from PIL import Image
from io import BytesIO
import base64
import cv2
import requests

from model import CNN
from models.classification.image_classification import predictLabel


model = CNN()
model.load_state_dict(torch.load('mnist_model.pt'), strict=False)
model.eval()

normalize = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

app = Flask(__name__)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

@app.route('/test', methods=['GET'])
def hello():
    print('hello!')
    return 'hello'

@app.route('/upload-image', methods=['POST'])
def uploadImage(file=None):
    if request.method == 'POST':
        pic_data = request.get_data().decode('utf-8')
    
    print('Upload Image Start')
    SAVED_IMAGE_PATH = 'image/origin_image.png'
    im = Image.open(BytesIO(base64.b64decode(pic_data)))
    im.save(SAVED_IMAGE_PATH ,'PNG')

    # 1) Image Classification Model
    category = predictLabel(im)
    
    im.resize((1024, 1024))
    im.save(SAVED_IMAGE_PATH ,'PNG')

    # 2) Kakao API 로 Text 검출 -> 가장 Height 큰 Text Return (임시)
    APP_KEY = '6a8361ba43e7da5ac8e1e90799287e0c'
    API_URL = 'https://dapi.kakao.com/v2/vision/text/ocr'
    headers = {'Authorization' : 'KakaAK {}'.format(APP_KEY)}
    
    kakao_img = cv2.imread(SAVED_IMAGE_PATH)
    peng_kakao_img = cv2.imencode(".png", kakao_img)[1]
    data = peng_kakao_img.tobytes()
    
    kakao_output = requests.post(API_URL, headers=headers, files={"image":data}).json()
    
    print(f"{bcolors.OKGREEN}SUCCESS: {bcolors.ENDC}", f"{bcolors.BOLD}{kakao_output}{bcolors.ENDC}" )
    
    return 'ok'
    

@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    _, result = model.forward(
        normalize(np.array(data['images'], dtype=np.uint8)).unsqueeze(0)).max(1)

    return str(result.item())


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=2431)