from PIL import Image
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from keras.models import Sequential
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD
import pickle as plk
import warnings
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")

TEST_IMAGE_PATH = './models/test'
DATA_PATH = './data/cifar-100-python'
MODEL_PATH = './models/classification/cifar_efficientnetb0_weights.h5'


'''
0) Pickle File -> Dicktionary
'''
def unpickle(file):
    with open(file, 'rb') as fo:
        myDict = plk.load(fo, encoding='latin1')
    return myDict


'''
1) Base64 Image -> Image File 로  복구
'''
def base64ImageToImageFile():
    return


'''
2) Image File -> numpy.d.array 로  변환
'''
def imageFileToNumpyArr(img):
    img = img.resize((224, 224))
    img.convert("RGB")
    npArr = np.asarray(img, dtype=np.float32) / 255

    data = npArr.tolist()
    data = [data]
    
    data = np.array(data)

    return data


'''
3) Output Label -> Category Text 로  변환
'''
def outputLabelToCategoryText(output):
    # output 은 (45, ) 의 numpy array
    target_idx = np.argmax(output)

    # Meta Data
    metaData = unpickle(DATA_PATH+'/meta')

    origin_label = [0, 2, 5, 8, 9, 10, 11, 12, 13, 16, 17, 20, 22, 25, 28, 35, 37, 
    39, 40, 41, 46, 48, 51, 53, 54, 57, 58, 61, 62, 68, 69, 70, 76, 
    81, 82, 83, 84, 85, 86, 87, 89, 90, 92, 94, 98]

    result_idx = origin_label[target_idx]
    result_category_name = metaData['fine_label_names'][result_idx]

    return result_category_name


'''
4) Load Model
    4-1) efficient net model load 및 학습된 가중치 load
    4-2) model complie
'''
def loadEfficientNetModel():
    n_classes=45
    input_shape = (224, 224, 3)
    efnb0 = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape, classes=n_classes)

    model = Sequential()
    model.add(efnb0)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='softmax'))

    # model.summary()
    optimizer = Adam(lr=0.0001)
    model.load_weights(MODEL_PATH)

    # model compiling
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

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

''' ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
5) Main Function - Predict Category
''' ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def predictLabel(targetImg):
    ''' 1) <image name>.png or <image name>.jpg 형태의 이미지를 numpy.nd.array 로 변환 '''
    target_img_arr = imageFileToNumpyArr(targetImg)

    ''' 2) Transfer Learning 된 EfficientNetB0 Model Load '''
    model = loadEfficientNetModel()
    predict_result = model.predict(target_img_arr)

    ''' 3) Category Predict '''
    category = outputLabelToCategoryText(predict_result[0])
    print(f"{bcolors.OKGREEN}SUCCESS: {bcolors.ENDC}", f"{bcolors.BOLD}{category}{bcolors.ENDC}" )

    return category


'''
[TEST Code]
5) 의 함수를 테스트 코드 형태로 작성, input image 가 존재하지 않음.
'''
testList = [
    # Image.open('../../test/test1.jpeg'),
    # Image.open('../../test/test2.jpeg'),
    # Image.open('../../test/test3.jpeg'),
    Image.open('/home/beobwoo/school/Flask/models/test/test3.jpg'),
    # Image.open(TEST_IMAGE_PATH + '/test3.jpg'),
    # Image.open(TEST_IMAGE_PATH + '/test4.jpg'),
    # Image.open(TEST_IMAGE_PATH + '/test5.jpg')
    ]

def testPredictLabel():
    return predictLabel(testList[0])

# for each in testList:
#     category = predictLabel(each)
#     print('[Predict Category Name] ', category)
