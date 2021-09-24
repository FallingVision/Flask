import torch.nn.functional as F
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch
import numpy as np
from torchvision import transforms
from flask import Flask, jsonify, request
from PIL import Image
from io import BytesIO
import base64
import cv2
import string
import argparse
# from models.textrecognition.textmodel import
import textrecognition.model as TextModel
from textrecognition.demo import demo
app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def text():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True,
                        help='path to image_folder which contains text images', default='./test_folder')
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int,
                        default=16, help='input batch size')
    parser.add_argument('--saved_model',
                        help="path to saved_model to evaluation",
                        default='./best_accuracy.pth')
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int,
                        default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=64,
                        help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=200,
                        help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true',
                        help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true',
                        help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str,
                        help='Transformation stage. None|TPS', default='TPS')
    parser.add_argument('--FeatureExtraction', type=str,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet', default='VGG')
    parser.add_argument('--SequenceModeling', type=str,
                        help='SequenceModeling stage. None|BiLSTM', default='BiLSTM')
    parser.add_argument('--Prediction', type=str,
                        help='Prediction stage. CTC|Attn', default='CTC')
    parser.add_argument('--num_fiducial', type=int, default=20,
                        help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='the size of the LSTM hidden state')

    opt = parser.parse_args()
    """ vocab / character number configuration """
    if opt.sensitive:
        # same with ASTER setting (use 94 char).
        opt.character = string.printable[:-6]

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    demo(opt)
    return 'hello'


@app.route('/test', methods=['GET'])
def hello():
    #MODEL_PATH = './best_accuracy.pth'
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    # optimizer = SGD(model, 0.1)

    # model = TextModel.Model(optimizer)
    # model.eval()
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True,
                        help='path to image_folder which contains text images', default='./test_folder')
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int,
                        default=16, help='input batch size')
    parser.add_argument('--saved_model',
                        help="path to saved_model to evaluation",
                        default='./best_accuracy.pth')
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int,
                        default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=64,
                        help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=200,
                        help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true',
                        help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true',
                        help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str,
                        help='Transformation stage. None|TPS', default='TPS')
    parser.add_argument('--FeatureExtraction', type=str,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet', default='VGG')
    parser.add_argument('--SequenceModeling', type=str,
                        help='SequenceModeling stage. None|BiLSTM', default='VGG')
    parser.add_argument('--Prediction', type=str,
                        help='Prediction stage. CTC|Attn', default='CTC')
    parser.add_argument('--num_fiducial', type=int, default=20,
                        help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='the size of the LSTM hidden state')

    opt = parser.parse_args()
    """ vocab / character number configuration """
    if opt.sensitive:
        # same with ASTER setting (use 94 char).
        opt.character = string.printable[:-6]

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    demo(opt)
    # text()
    return 'hello'


@app.route('/upload-image', methods=['POST'])
def uploadImage(file=None):
    if request.method == 'POST':
        pic_data = request.get_data().decode('utf-8')

    im = Image.open(BytesIO(base64.b64decode(pic_data)))
    im.save('image/origin_image.png', 'PNG')

    # 이미지 사이즈 변환
    return 'ok'


@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    _, result = model.forward(
        normalize(np.array(data['images'], dtype=np.uint8)).unsqueeze(0)).max(1)

    return str(result.item())


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=2431)
