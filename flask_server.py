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

import requests
import json
#from model import CNN
from models.classification.image_classification import predictLabel

LIMIT_PX = 1024
LIMIT_BYTE = 1024*1024
LIMIT_BOX = 40

@app.route('/test', methods=['GET'])
def test():
    #MODEL_PATH = './best_accuracy.pth'
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    # optimizer = SGD(model, 0.1)

    # model = TextModel.Model(optimizer)
    # model.eval()
    # -*- coding: utf-8 -*-
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True,
                        help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int,
                        default=192, help='input batch size')
    parser.add_argument('--saved_model',
                        help="path to saved_model to evaluation", default='./best_accuracy.pth')
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int,
                        default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=64,
                        help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=200,
                        help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz'
                        '가각간갇갈감갑값갓강갖같갚갛개객걀걔거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀귓규균귤그극근글긁금급긋긍'
                        '기긴길김깅깊까깍깎깐깔깜깝깡깥깨꺼꺾껌껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꾼꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냇냉냐냥너넉넌널넓넘넣네넥넷'
                        '녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐댓더덕던덜덟덤덥덧덩덮데델도독돈돌돕돗동돼되된두둑둘둠둡둥'
                        '뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯'
                        '료루룩룹룻뤄류륙률륭르른름릇릎리릭린림립릿링마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몬몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭘뭣므미'
                        '민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벨벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브븐블비빌빔빗빚빛빠빡빨빵빼뺏뺨뻐뻔뻗'
                        '뼈뼉뽑뿌뿐쁘쁨사삭산살삶삼삿상새색샌생샤서석섞선설섬섭섯성세섹센셈셋셔션소속손솔솜솟송솥쇄쇠쇼수숙순숟술숨숫숭숲쉬쉰쉽슈스슨슬슴습슷승시식신싣실싫심십싯싱싶싸'
                        '싹싼쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓴쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액앨야약얀얄얇양얕얗얘어억언얹얻얼엄업없엇엉엊엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온'
                        '올옮옳옷옹와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡잣장잦재쟁쟤저적전절젊점접젓정젖제젠젯져조'
                        '족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쩔쩜쪽쫓쭈쭉찌찍찢차착찬찮찰참찻창찾채책챔챙처척천철첩첫청체쳐초촉촌촛총촬최'
                        '추축춘출춤춥춧충취츠측츰층치칙친칠침칫칭카칸칼캄캐캠커컨컬컴컵컷케켓켜코콘콜콤콩쾌쿄쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱턴털텅테텍텔템토톤톨톱통퇴투툴툼퉁'
                        '튀튜트특튼튿틀틈티틱팀팅파팎판팔팝패팩팬퍼퍽페펜펴편펼평폐포폭폰표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈협형혜호혹혼홀홈홉홍'
                        '화확환활황회획횟횡효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘?!', help='character label')
    parser.add_argument('--sensitive', action='store_true',
                        help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true',
                        help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str,
                        required=True, help='Transformation stage. None|TPS', default='TPS')
    parser.add_argument('--FeatureExtraction', type=str,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet', default='ResNet')
    parser.add_argument('--SequenceModeling', type=str,
                        required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str,
                        required=True, help='Prediction stage. CTC|Attn', default='Attn')
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
    lst=[]
    with open('./result.txt', 'rb') as f:
        # line_num = 1
        # line_data= f.readline()
        # while line_data:
        predict = f.readlines()

        #     lst.append(predict)
        #     line_num+=1
        #predict = f.readline()
        #lst.append(predict)
    # text()
    for i in range(0,len(predict)):
        predict[i]=predict[i].decode('utf-8')
        predict[i]=predict[i][:-1]
    lst= [[o] for o in predict]
    return lst


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

# @app.route('/upload-image', methods=['POST'])
# def uploadImage(file=None):
#     if request.method == 'POST':
#         pic_data = request.get_data().decode('utf-8')

#     SAVED_IMAGE_PATH = 'image/origin_image.jpg'
#     im = Image.open(BytesIO(base64.b64decode(pic_data)))
#     im.save(SAVED_IMAGE_PATH)

#     # 1) Image Classification Model
#     category = predictLabel(im)

#     # 2) Kakao API 로 Text 검출 -> 가장 Height 큰 Text Return (임시)
#     APP_KEY = 'af23d4b5ea3248412227a7bce9609752'
#     API_URL = 'https://dapi.kakao.com/v2/vision/text/ocr'
#     headers = {'Authorization': 'KakaoAK {}'.format(APP_KEY)}

#     resize_impath = kakao_ocr_resize(SAVED_IMAGE_PATH)
#     kakao_img = cv2.imread(resize_impath)
#     peng_kakao_img = cv2.imencode(".jpg", kakao_img)[1]
#     data = peng_kakao_img.tobytes()

#     kakao_output = requests.post(
#         API_URL, headers=headers, files={"image": data}).json()
        
 

#     for i in range(1, len(kakao_output['result'])):
#         if len(kakao_output['result'][i]['recognition_words'][0]) == 0 :
#             kakao_output['result'].pop(i)


#     extract_text_list = [o['recognition_words'] for o in kakao_output['result']]

#     print(f"{bcolors.OKGREEN}SUCCESS: {bcolors.ENDC}",
#           f"{bcolors.BOLD}{extract_text_list}{bcolors.ENDC}")
    
#     temp_idx = 0
    
#     if len(kakao_output['result']) > 0 and len(kakao_output['result'][0]) > 0 :
#         temp = kakao_output['result'][0]['boxes'][2][1] - \
#         kakao_output['result'][0]['boxes'][1][1] 
        
#         product_name = kakao_output['result'][0]['recognition_words'][0]
    
#         for i in range(1, len(kakao_output['result'])):
#             temp2 = kakao_output['result'][i]['boxes'][2][1] - \
#                 kakao_output['result'][i]['boxes'][1][1]
#             if(temp < temp2):
#                 temp = temp2
#                 temp_idx = i
                
#         data = json.dumps({"category": category, "main_text_idx": temp_idx, "text_list": extract_text_list, "error": False})
#     else :
#         data = json.dumps({"category": category, "main_text_idx": temp_idx, "text_list": [], "error": True})
    
#     return data

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
    print(f'--------------------------------------')
    print(f'------{bcolors.OKGREEN}kakao recognition words{bcolors.ENDC}---------')
    print(f'--------------------------------------')
    for i in range(0,len(kakao_output['result'])):  
        print('recognition_words:      ',kakao_output['result'][i]['recognition_words'][0])

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
    
    
    image = Image.open(resize_impath)

    SAVED_CROP_PATH='image/crop_image/origin_image.jpg'
    #image = Image.open(RESIZED_IMAGE_PATH)
    for i in range(0,len(kakao_output['result'])):
        area= (kakao_output['result'][i]['boxes'][0][0],kakao_output['result'][i]['boxes'][0][1],
        max(kakao_output['result'][i]['boxes'][1][0],kakao_output['result'][i]['boxes'][2][0]),
        max(kakao_output['result'][i]['boxes'][3][1],kakao_output['result'][i]['boxes'][2][1]))
        crop_img=image.crop(area)

        crop_file_name=SAVED_CROP_PATH[:-4]+'_'+str(i)+'.jpg'
        crop_img.save(crop_file_name)
    text_list=test()
    print(text_list)
    data = json.dumps({"category": category, "main_text_idx": 1, "text_list": text_list, "error": False})
    return data

@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    _, result = model.forward(
        normalize(np.array(data['images'], dtype=np.uint8)).unsqueeze(0)).max(1)

    return str(result.item())


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=2431)
