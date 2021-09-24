import torch
import numpy as np
from torchvision import transforms
from flask import Flask, jsonify, request
from PIL import Image
from io import BytesIO
import base64

from model import CNN


model = CNN()
model.load_state_dict(torch.load('mnist_model.pt'), strict=False)
model.eval()

normalize = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

app = Flask(__name__)

@app.route('/test', methods=['GET'])
def hello():
    print('hello!')
    return 'hello'

@app.route('/upload-image', methods=['POST'])
def uploadImage(file=None):
    if request.method == 'POST':
        pic_data = request.get_data().decode('utf-8')
    
    im = Image.open(BytesIO(base64.b64decode(pic_data)))
    im.save('image/origin_image.png' ,'PNG')

    # 이미지 사이즈 변환

    return 'ok'
    

@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    _, result = model.forward(
        normalize(np.array(data['images'], dtype=np.uint8)).unsqueeze(0)).max(1)

    return str(result.item())


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=2431)