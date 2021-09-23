from PIL import Image
import numpy as np
from tensorflow import keras

'''
1) Base64 Image -> Image File 로  복구
'''

'''
2) Image File -> numpy.d.array 로  변환
'''
def imageFileToNumpyArr(img):
    # print('origin', img.size)
    img = img.resize((224, 224))
    # print('resize', img.size)
    npArr = np.array(img)
    # print('np', npArr.shape)
    
    
    data = npArr.tolist()
    data = [data]
    
    data = np.array(data)
    data = data/255.0
    # print(type(data), len(data))
    
    data = data.reshape(len(data), 3, 224, 224).transpose(0, 2, 3, 1)
    # print(data.shape)
    
    return 'a'

'''
3) Output Label -> Category Text 로  변환
'''
def outputLabelToCategoryText(output):
    # output 은 (45, ) 의 numpy array
    return
    

testImage1 = Image.open('../../test/test1.jpeg')
imageFileToNumpyArr(testImage1)

# efnb0 = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape, classes=n_classes)

# model = Sequential()
# model.add(efnb0)
# model.add(GlobalAveragePooling2D())
# model.add(Dropout(0.2))
# model.add(Dense(n_classes, activation='softmax'))

# model.summary()

# optimizer = Adam(lr=0.0001)

# # model compiling
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# model = keras.models.load_weights('./cifar_efficientnetb0_weights.h5')
# model.summary()