from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input as resnet50_v2_preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_v3_preprocess_input

MODELS = {
    'mobilenet_v2': (MobileNetV2, mobilenet_v2_preprocess_input),
    'resnet50_v2': (ResNet50V2, resnet50_v2_preprocess_input),
    'vgg16': (VGG16, vgg16_preprocess_input),
    'inception_v3': (InceptionV3, inception_v3_preprocess_input)
}

# Define minimum input sizes for each model
INPUT_SIZES = {
    'mobilenet_v2': [32, 64, 128, 256],
    'resnet50_v2': [32, 64, 128, 256],
    'vgg16': [32, 64, 128, 256],
    'inception_v3': [75, 128, 256]
}
