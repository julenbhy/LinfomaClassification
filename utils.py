import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_generators(preprocessing_function, dataset_path, batch_size, size, n_plots=0):
    """
    Creates data generators for training, validation, and testing.

    Args:
        preprocessing_function: Function to apply for preprocessing.
        dataset_path: Path to the split dataset directory.
        batch_size: Batch size.
        size: Image size.
        prints: Ammout of sample images to print

    Returns:
        train_generator: Training data generator.
        validation_generator: Validation data generator.
        test_generator: Test data generator.
    """

    train_datagen = ImageDataGenerator(
        fill_mode='wrap',
        preprocessing_function=preprocessing_function
    )

    print("Loading train data")
    train_generator = train_datagen.flow_from_directory(
        dataset_path + '/train',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        target_size=(size, size)
    )
    
    val_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
    
    print("Loading validation data")
    validation_generator = val_datagen.flow_from_directory(
        dataset_path + '/val',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        target_size=(size, size)
    )

    print("Loading test data")
    test_generator = val_datagen.flow_from_directory(
        dataset_path + '/test',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        target_size=(size, size)
    )
    
    # Get a test batch from the generator to ensure images are loaded correctly
    images, labels = train_generator.__next__()
    print('Images shape:', images.shape)
    
    for i in range(n_plots):
        f = plt.figure(figsize=(20, 20))
        f.add_subplot(1, 3, 1)
        plt.axis('off')
        plt.title('Label: ' + str(np.argmax(labels[i])))
        plt.imshow(images[i])
    plt.show(block=True)

    return train_generator, validation_generator, test_generator 


def build_model(model_fn, input_shape, num_classes):
    """
    Builds and compiles a CNN model.

    Args:
        model_fn: Function to create the base model (e.g., ResNet50).
        input_shape: Shape of the input images.
        num_classes: Number of classes for classification.

    Returns:
        model: Compiled CNN model.
    """

    # 
    from tensorflow.keras.layers import GlobalMaxPooling2D, Dense
    from tensorflow.keras.models import Model
    
    # Create the base model from the pre-trained model
    base_model = model_fn(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False

    # Add a classification head
    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dense(32, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    metrics = ['accuracy', 'Precision', 'Recall', 'AUC']

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

    return model



from PIL import Image
import os
import numpy as np

def is_valid_image(file_path):
    try:
        i = Image.open(file_path)
        valid = True
    except:
        valid = False
    return valid

def detect_corrupted_images(dataset_path):
    corrupted_images = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not is_valid_image(file_path):
                corrupted_images.append(file_path)
    
    print('Number of corrupted images:', len(corrupted_images))
    return corrupted_images
