import streamlit as st
import tensorflow as tf
import numpy as np
import os
from os.path import exists
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Machine learning
tf.config.run_functions_eagerly(True)

st.set_page_config(
    page_title="Implenment model",
    page_icon="🧊",
    layout="wide"
)

# Load the model Cache
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(os.path.join('model', 'hotaEfficientNetV2S_super_sigmoid_224x224_v7.keras'))
    return model

# Load model and init AI
with st.spinner('Loading...'):
    model = load_model()

# Helper function
# torchvision.transforms.Normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
def torchNormalize(data: np.ndarray, mean: list[float], std: list[float], inplace=False) -> np.ndarray:
    # Inplace operation check
    if not inplace:
        data = np.copy(data).astype("float32")

    # Convert mean and std to NumPy arrays with appropriate shape
    mean = np.asarray(mean, dtype=data.dtype)
    std = np.asarray(std, dtype=data.dtype)

    # Perform normalization
    return (data - data.mean()) / data.std() * std + mean
def convertBinVec2LabelList(binVec):
    labelValue = ["Hồ Gươm", "Hồ Tây", "Tháp rùa", "Cầu Thê Húc", "Bưu Điện", "Vườn Hoa", "Chùa Trấn Quốc", "Đền Quán Thánh", "Khách Sạn", "Công Viên Nước"]
    labels = []
    for i in range(len(binVec)):
        if binVec[i] > 0.5:
            labels.append(labelValue[i])
    return labels
def applyLogicLabel(label):
    listAddPoint = [
        [0, 2, 3, 4, 5, 8],
        [1, 4, 5, 6, 7, 8, 9],
    ]
    # Caculate point
    pointTemp = [0] * len(listAddPoint)
    for i in range(len(label)):
        if int(label[i]) == 0:
            continue
        for j in range(len(listAddPoint)):
            if i in listAddPoint[j]:
                pointTemp[j] += 1
    # Find max
    pointJ, indexJ = 0, 0
    for j in range(len(pointTemp)):
        if pointJ < pointTemp[j]:
            pointJ = pointTemp[j]
            indexJ = j
    # Flter label
    labelRes = [0]*len(label)
    for i in range(len(label)):
        if int(label[i]) == 1 and (i in listAddPoint[indexJ]):
            labelRes[i] = 1
    # Add location
    labelRes[indexJ] = 1
    
    return labelRes
def decodePredict2BinVec(predict):
    # Convert predict to binary vector
    binVec = []
    for i in range(len(predict)):
        if predict[i] > 0.5:
            binVec.append(1)
        else:
            binVec.append(0)
    return binVec
def resize_image_reduce_size(image, size_image_model):
    min_size = min(image.shape[0], image.shape[1])
    radio_h = image.shape[0] / min_size
    radio_w = image.shape[1] / min_size
    return cv2.resize(image, (int(radio_w * size_image_model), int(radio_h * size_image_model)))
def crop2nimage(image, nimage = 3):
    # Size of bigest square image
    sizeSquare = min(image.shape[0], image.shape[1])

    # Rotate image if image is portrait
    isRotate = False
    if image.shape[0] > image.shape[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        isRotate = True

    # Crop n image
    cropImages = []
    widthBetween = (image.shape[1] - sizeSquare) / (nimage - 1)
    for i in range(nimage):
        x1 = int(i * widthBetween)
        x2 = int(x1 + sizeSquare)
        cropImages.append(image[:, x1:x2])

    # Rotate image back
    if isRotate:
        for i in range(len(cropImages)):
            cropImages[i] = cv2.rotate(cropImages[i], cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Convert color to RGB
    # for i in range(len(cropImages)):
        # cropImages[i] = cv2.cvtColor(cropImages[i], cv2.COLOR_BGR2RGB)
        
    return cropImages


# Main
with st.container():
    st.title('AI model')
    st.write('This is a simple AI model that can predict the image')

    # Image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:        
        image = tf.io.decode_image(uploaded_file.getvalue(), channels=3).numpy()
        image = resize_image_reduce_size(image, 224)
        image = crop2nimage(image, 3)[1]
        # image = torchNormalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        imageNol = tf.image.per_image_standardization(image).numpy()
        prediction = model.predict(np.expand_dims(imageNol, axis=0))[0]

        # 2 columns layout, with size ratio 2:1
        col1, col2 = st.columns([1, 1])
        with col1:
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
        with col2:
            st.write('Prediction: ', str(prediction))
            st.write('Label: ', convertBinVec2LabelList(applyLogicLabel(decodePredict2BinVec(prediction))))