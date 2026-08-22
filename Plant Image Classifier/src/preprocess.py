from utils import *;

with ZipFile("plantvillage-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall()

print(os.listdir("plantvillage dataset"))

print(len(os.listdir("plantvillage dataset/segmented")))
print(os.listdir("plantvillage dataset/segmented")[0:5])

print(len(os.listdir("plantvillage dataset/segmented")))
print(os.listdir("plantvillage dataset/segmented")[:5])

print(len(os.listdir("plantvillage dataset/segmented")))
print(os.listdir("plantvillage dataset/segmented")[:5])

print(len(os.listdir("plantvillage dataset/color/Grape_healthy")))
print(os.listdir("plantvillage dataset/segmented")[:5])

# Dataset Path
base_dir = 'plantvillage dataset/color'

image_path = ''

# Read the image
img = mpimg.imread(image_path)

print(img.shape)
#Displat the image
plt.imshow(img)
plt.axis('off') # Turn off axis numbers
plt.show()

# Image Parameters
img_size = 224
batch_size = 32

# Image Data Generators
data_gen = ImageDataGenerator(
    rescale = 1./255,
    validation_spilt = 0.2 # Use 20% of data for validation
)

# Train Generator
train_generator = data_gen.flow_from_directory(
    base_dir,
    target_size = (img_size, img_size),
    batch_size = batch_size,
    subset = 'training',
    clas_mode = 'catagorical'
)

#validation Generator
validation_generator = data_gen.flow_from_directory(
    base_dir,
    target_size = (img_size, img_size),
    batch_size = batch_size,
    subset = 'validation',
    
)