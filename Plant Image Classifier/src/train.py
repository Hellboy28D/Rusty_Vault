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
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[predicted_class_index]
    return predicted_class_name

# Create a mapping from class indices to class names
class_indices = {v: k for k, v in train_generator.class_indices.items()}
class_indices

# saving the class names as json file
json.dump(class_indices, open('class_indices.json', 'w'))
 # Example Usage
 #image_path = ''
 #image_path = ''
image_path = ''
predicted_class_name = predict_image_class(model, image_path, class_indices)

#Output the result
print("predicted Class Name:", predicted_class_name)