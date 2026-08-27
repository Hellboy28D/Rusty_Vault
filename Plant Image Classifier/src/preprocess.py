from pickletools import optimize
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
    class_mode = 'catagorical'
)

# Validation Generator
validation_generator = data_gen.flow_from_directory(
    base_dir,
    target_size = (img_size, img_size),
    batch_size = batch_size,
    subset = 'validation',
    class_mode = 'categorical'
)

# Model Defination
model = models.Sequential()

model.add(layers.conv2D(32, (3,3), activation = 'relu', input_shape=(img_size, img_size, 3)))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.conv2D(62, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(train_generator.num_classes, activation= 'softmax'))

# model summary
model.summary()

# Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
)

# Training the Model
history = model.fit(
    train_generator,
    steps_per_epoch= train_generator.samples // batch_size,
    epochs = 5,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size # Validation
) 

# Model Evaluation
print("Evaluation model...")
val_boss, val_accuracy = model.evaluate(validation_generator, steps = validation_generator // batch_size )
print(f"Validationg Accuracy: {val_accuracy * 100: 2f} %")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylable('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylable('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.show()