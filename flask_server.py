import torch
import numpy as np
from torchvision import transforms
from flask import Flask, jsonify, request
from PIL import Image
from io import BytesIO
import base64
import cv2
import requests
import json
from model import CNN
from models.classification.image_classification import predictLabel

LIMIT_PX = 1024
LIMIT_BYTE = 1024*1024
LIMIT_BOX = 40


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


def kakao_ocr_resize(image_path: str):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    if LIMIT_PX < height or LIMIT_PX < width:
        ratio = float(LIMIT_PX)/max(height, width)
        image = cv2.resize(image, None, fx=ratio, fy=ratio)
        height, width, _ = height, width, _ = image.shape

        image_path = "{}_resized.jpg".format(image_path)
        cv2.imwrite(image_path, image)
        return image_path
    return None


@app.route('/test', methods=['GET'])
def hello():
    print('hello!')
    return 'hello'


@app.route('/upload-image', methods=['POST'])
def uploadImage(file=None):
    if request.method == 'POST':
        pic_data = request.get_data().decode('utf-8')

    SAVED_IMAGE_PATH = 'image/origin_image.jpg'
    im = Image.open(BytesIO(base64.b64decode(pic_data)))
    im.save(SAVED_IMAGE_PATH)

    # 1) Image Classification Model
    category = predictLabel(im)

    # 2) Kakao API 로 Text 검출 -> 가장 Height 큰 Text Return (임시)
    APP_KEY = 'af23d4b5ea3248412227a7bce9609752'
    API_URL = 'https://dapi.kakao.com/v2/vision/text/ocr'
    headers = {'Authorization': 'KakaoAK {}'.format(APP_KEY)}

    resize_impath = kakao_ocr_resize(SAVED_IMAGE_PATH)
    kakao_img = cv2.imread(resize_impath)
    peng_kakao_img = cv2.imencode(".jpg", kakao_img)[1]
    data = peng_kakao_img.tobytes()

    kakao_output = requests.post(
        API_URL, headers=headers, files={"image": data}).json()
        
 

    for i in range(1, len(kakao_output['result'])):
        if len(kakao_output['result'][i]['recognition_words'][0]) == 0 :
            kakao_output['result'].pop(i)


    extract_text_list = [o['recognition_words'] for o in kakao_output['result']]

    print(f"{bcolors.OKGREEN}SUCCESS: {bcolors.ENDC}",
          f"{bcolors.BOLD}{extract_text_list}{bcolors.ENDC}")
    
    temp_idx = 0
    
    if len(kakao_output['result']) > 0 and len(kakao_output['result'][0]) > 0 :
        temp = kakao_output['result'][0]['boxes'][2][1] - \
        kakao_output['result'][0]['boxes'][1][1] 
        
        product_name = kakao_output['result'][0]['recognition_words'][0]
    
        for i in range(1, len(kakao_output['result'])):
            temp2 = kakao_output['result'][i]['boxes'][2][1] - \
                kakao_output['result'][i]['boxes'][1][1]
            if(temp < temp2):
                temp = temp2
                temp_idx = i
                
        data = json.dumps({"category": category, "main_text_idx": temp_idx, "text_list": extract_text_list, "error": False})
    else :
        data = json.dumps({"category": category, "main_text_idx": temp_idx, "text_list": [], "error": True})

    return data


@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    _, result = model.forward(
        normalize(np.array(data['images'], dtype=np.uint8)).unsqueeze(0)).max(1)

    return str(result.item())


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=2431)
