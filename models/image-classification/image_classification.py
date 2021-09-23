from PIL import Image
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from keras.models import Sequential, load_model
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD
import pickle as plk
import warnings
warnings.filterwarnings("ignore")

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
    DATA_PATH = '../../data/cifar-100-python'
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
    model.load_weights('./cifar_efficientnetb0_weights.h5')

    # model compiling
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

'''
5) Predict Category
'''
def predictLabel(img):
    target_img_arr = imageFileToNumpyArr(img)
    model = loadEfficientNetModel()
  
    predict_result = model.predict(target_img_arr)
    category = outputLabelToCategoryText(predict_result[0])

    return category


'''
[TEST Code]
'''
testList = [
    # Image.open('../../test/test1.jpeg'),
    # Image.open('../../test/test2.jpeg'),
    # Image.open('../../test/test3.jpeg'),
    Image.open('../../test/test1.jpeg'),
    Image.open('../../test/test5.jpeg'),
    Image.open('../../test/test6.jpeg'),
    Image.open('../../test/test7.jpeg')
    ]

for each in testList:
    category = predictLabel(each)
    print('[Predict Category Name] ', category)
''' ''' ''' ''' ''' ''' ''' ''' ''' '''