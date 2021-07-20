
# pytorch 모델을 서빙
import torch
import numpy as np
from torchvision import transforms
from flask import Flask, jsonify, request, json
import base64
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
# from model import CNN


# model = CNN()
# model.load_state_dict(torch.load('mnist_model.pt'), strict=False)
# model.eval()
# # normalize = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

app = Flask(__name__)


# @app.route('/inference', methods=['POST'])
# def inference():
#     data = request.json
#     _, result = model.forward(
#         normalize(np.array(data['images'], dtype=np.uint8)).unsqueeze(0)).max(1)
#     print(str(result.item()))
#     return str(result.item())


@app.route('/test', methods=['GET'])
def test():
    # params = request.args.get('image')
    # print(type(params))
    # print('aaaaaaaaaa', params)
    # if len(params) <= 0:
    # return 'No parameter'

    # test 이미지 확인
    # with open('./test_image.jpg', 'rb') as img:
    #data = base64.b64encode(img.read())
    # print(data)
    # base64 이미지 열기
    # im = Image.open(BytesIO(base64.b64decode(params)))
    #im = Image.open(BytesIO(base64.b64decode(data)))
    # test이미지 저장
    # im.save('image.png', 'PNG')

    # 이미지 크기 출력
    # print(im.size)
    # 이미지 show
    # plt.imshow(img)

    # pt 파일이 들어갈곳
    category = 'picture'
    text = 'mug'
    response = json.dumps({'category': category, 'text': text})
    print(response)
    # return im
    return response


@app.route('/apitest', methods=['GET'])
def apitest():
    category = 'clothes'
    text = 'test'
    response = json.dumps({'category': category, 'text': text})
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
