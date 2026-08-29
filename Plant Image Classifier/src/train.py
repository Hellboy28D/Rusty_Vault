from utils import *;
from preprocess import *;


# *** Function to Load Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values tp [0,1]
    img_array =  img_array.astype('float32') / 255
    return img_array

# Function to predict the class of an Image
def predict_image_class(model, image_path, class_indices):
    