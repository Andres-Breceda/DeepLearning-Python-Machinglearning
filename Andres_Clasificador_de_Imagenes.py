import os
import shutil


data =  r"C:\Users\Andre\Downloads\dogs-vs-cats (1)\train\train"

# Crear carpetas si no existen
os.makedirs(os.path.join(data, 'cat'), exist_ok=True)
os.makedirs(os.path.join(data, 'dog'), exist_ok=True)

# Mover imágenes a sus respectivas carpetas
for filename in os.listdir(data):
    src = os.path.join(data, filename)
    if os.path.isfile(src):  # Solo mover archivos, no carpetas
        if filename.startswith('cat'):
            shutil.move(src, os.path.join(data, 'cat', filename))
        elif filename.startswith('dog'):
            shutil.move(src, os.path.join(data, 'dog', filename))

            #Visualizacion de imagenes

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

data = "/workspaces/machine-learning-python-linear-regression/dogs-vs-cats (1)/train/train"

# Mostrar 9 imágenes de gatos
cat_images = [f for f in os.listdir("/workspaces/machine-learning-python-linear-regression/dogs-vs-cats (1)/train/train/cat") if 'cat' in f][:9]
plt.figure(figsize=(10, 10))
for i, img_name in enumerate(cat_images):
    img_path = os.path.join("/workspaces/machine-learning-python-linear-regression/dogs-vs-cats (1)/train/train/cat", img_name)
    img = mpimg.imread(img_path)
    plt.subplot(3, 3, i+1)
    plt.imshow(img)
    plt.title("Cat")
    plt.axis('off')
plt.show()

# Repetir con imágenes de perros
dog_images = [f for f in os.listdir("/workspaces/machine-learning-python-linear-regression/dogs-vs-cats (1)/train/train/dog") if 'dog' in f][:9]
plt.figure(figsize=(10, 10))
for i, img_name in enumerate(dog_images):
    img_path = os.path.join("/workspaces/machine-learning-python-linear-regression/dogs-vs-cats (1)/train/train/dog", img_name)
    img = mpimg.imread(img_path)
    plt.subplot(3, 3, i+1)
    plt.imshow(img)
    plt.title("Dog")
    plt.axis('off')
plt.show()

import os
import cv2

#Gatos y perros -- Cambiar cat por dog para transformar las dos carpetas
input_folder = r"C:\Users\Andre\Downloads\dogs-vs-cats (1)\train\train\cat"
output_folder = input_folder  # Mismo folder
target_size = (180, 180)

# Procesar cada imagen
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"No se pudo leer la imagen: {filename}")
            continue

        h, w = img.shape[:2]

        if (w, h) != target_size:
            resized_img = cv2.resize(img, target_size)
            print(f"Redimensionando: {filename} de ({w}, {h}) a {target_size}")
        else:
            resized_img = img
            print(f"Ya está en 640x640: {filename}")

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, resized_img)


# Verificando que funcione Targeta grafica con tensorflow

import tensorflow as tf

# Mostrar versión de TensorFlow
print("Versión de TensorFlow:", tf.__version__)

# Verificar dispositivos disponibles
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU detectada:")
    for gpu in gpus:
        print(gpu)
else:
    print("❌ No se detectó GPU. Está usando CPU.")


    from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(input_shape=(180, 180, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(units=4096, activation="relu"))
model.add(Dense(units=4096, activation="relu"))
model.add(Dense(units=2, activation="softmax"))


import numpy as np
fake_input = np.zeros((1, 180, 180, 3))
output = model.predict(fake_input)
print("Shape after Flatten:", output.shape)  # Esto da (1, 18432)

model.summary()

from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    '/mnt/c/Users/Andre/Downloads/dogs-vs-cats (1)/train/train',
    target_size=(180, 180),
    batch_size=32,
    class_mode='sparse',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    '/mnt/c/Users/Andre/Downloads/dogs-vs-cats (1)/train/train',
    target_size=(180,180),
    batch_size=32,
    class_mode='sparse',
    subset='validation'
)

import time
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, mode='max')
early = EarlyStopping(monitor='val_accuracy', patience=5)

start_time = time.time()  # Marca el inicio

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=[checkpoint, early]
)

end_time = time.time()  # Marca el fin

elapsed_time = end_time - start_time  # Tiempo total en segundos

print(f"Tiempo total de entrenamiento: {elapsed_time:.2f} segundos")
print(f"Tiempo total de entrenamiento: {elapsed_time / 60:.2f} minutos")

# Optimizando modelo


# Detiene el entrenamiento si la validación no mejora tras 5 epochs
early = EarlyStopping(
    monitor='val_accuracy', 
    patience=5, 
    restore_best_weights=True, 
    verbose=1
)


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, mode='max')
early = EarlyStopping(monitor='val_accuracy', patience=5)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=[checkpoint, early]
)

## Mejor modelo Optimizado

#Por lo visto el modelo el modelo no tiene buena precision ni aun en la optimizacion. Mejorando algunas caracterizticas a la hora de crear el 
#el modelo podemos mejorar su acuracy. Existen redes neuronales CNN, mas robustos a la hora de hacer un clasificador de imagen como 
#las redes convulocionales.

import os

os.makedirs("modelo_final", exist_ok=True)

#Guardamos el modelo

from tensorflow.keras.models import load_model
best_model = load_model("best_model.h5")

best_model.save("modelo_final/modelo_entrenado.h5")


import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.utils import load_img, img_to_array
#import matplotlib.pyplot as plt

# ==== CONFIGURACIÓN ====
data_dir = data_dir = r"/mnt/c/Users/Andre/Downloads/dogs-vs-cats (1)/train/train" # Cambia esto por el path a tu dataset
img_height = 180
img_width = 100
batch_size = 16  # Puedes bajarlo si usas CPU o poca RAM

# ==== CARGA Y PREPROCESAMIENTO DE DATOS ====
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Clases detectadas:", class_names)

# ==== MODELO CNN ====
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names))  # salida = número de clases
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, mode='max')
early = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[checkpoint, early]
)

# Guardar modelo

checkpoint = ModelCheckpoint("ModelOptimizado.h5", monitor='val_accuracy', save_best_only=True, mode='max')

model.save("modelo_final/ModelOptimizad.h5")

# Metricas

from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# 1. Cargar modelo
model = load_model("modelo_final/ModelOptimizad.h5")

# 2. Obtener etiquetas verdaderas y predicciones
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# 3. Calcular matriz de confusión
cm = confusion_matrix(y_true, y_pred)
print("Matriz de confusión:")
print(cm)

# 4. Reporte de clasificación
print("Reporte de clasificación:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Como vemos el modelo lo hace razonablemente bien

