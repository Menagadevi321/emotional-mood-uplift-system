import os
import time
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pygame  # For local mp3 playback

# ========== Parameters ==========
img_width, img_height = 48, 48
batch_size = 64
epochs = 50

train_dir = r'D:/emotion dataset 4/images/train'
test_dir = r'D:/Emotion dataset 1/test'

# ========== 1. Preprocess Data ==========
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='categorical',
    subset='training', color_mode='grayscale'
)

val_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='categorical',
    subset='validation', color_mode='grayscale'
)

# ========== 2. Build Model ==========
model = Sequential([
    Input(shape=(img_width, img_height, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ========== 3. Train Model ==========
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

model.save("emotion_model.keras")

# ========== Plot Accuracy ==========
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Training & Validation Accuracy")
plt.legend()
plt.show()

# ========== 4. Evaluate on Test Set ==========
if os.path.exists(test_dir):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=(img_width, img_height),
        batch_size=1, class_mode='categorical',
        shuffle=False, color_mode='grayscale'
    )

    loss, accuracy = model.evaluate(test_generator)
    print(f"\nTest Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    y_pred = np.argmax(model.predict(test_generator), axis=1)
    print("\nClassification Report:\n")
    print(classification_report(test_generator.classes, y_pred,
                                target_names=test_generator.class_indices.keys()))
    print("Confusion Matrix:")
    print(confusion_matrix(test_generator.classes, y_pred))

# ========== 5. Local MP3 Integration ==========
print("\n[INFO] Setting up local MP3 playback...")
pygame.mixer.init()

emotion_music_files = {
    'angry': [
        r"C:/Users/AUSU/Desktop/songs/angry/calm1.mp3",
        r"C:/Users/AUSU/Desktop/songs/angry/calm2.mp3",
        r"C:/Users/AUSU/Desktop/songs/angry/calm3.mp3",
        r"C:/Users/AUSU/Desktop/songs/angry/calm4.mp3",
        r"C:/Users/AUSU/Desktop/songs/angry/calm5.mp3"
    ],
    'fearful': [
        r"C:/Users/AUSU/Desktop/songs/fear/mt1.mp3",
        r"C:/Users/AUSU/Desktop/songs/fear/mt2.mp3",
        r"C:/Users/AUSU/Desktop/songs/fear/mt3.mp3",
        r"C:/Users/AUSU/Desktop/songs/fear/mt4.mp3",
        r"C:/Users/AUSU/Desktop/songs/fear/mt5.mp3"
    ],
    'happy': [
        r"C:/Users/AUSU/Desktop/songs/happy/h4.mp3",
        r"C:/Users/AUSU/Desktop/songs/happy/h5.mp3",
        r"C:/Users/AUSU/Desktop/songs/happy/h1.mp3",
        r"C:/Users/AUSU/Desktop/songs/happy/h2.mp3",
        r"C:/Users/AUSU/Desktop/songs/happy/h3.mp3"
    ],
    'neutral': [
        r"C:/Users/AUSU/Desktop/songs/neutral/song3.mp3",
        r"C:/Users/AUSU/Desktop/songs/neutral/song1.mp3",
        r"C:/Users/AUSU/Desktop/songs/neutral/song2.mp3"
    ],
    'sad': [
        r"C:/Users/AUSU/Desktop/songs/sad/sad4.mp3",
        r"C:/Users/AUSU/Desktop/songs/sad/sad1.mp3",
        r"C:/Users/AUSU/Desktop/songs/sad/sad2.mp3",
        r"C:/Users/AUSU/Desktop/songs/sad/sad3.mp3"
    ]
}

def play_emotion_music(emotion_label):
    music_files = emotion_music_files.get(emotion_label)
    if not music_files:
        print(f"[MP3] No music files found for emotion: {emotion_label}")
        return
    
    # Choose a random song if multiple
    music_file = random.choice(music_files) if isinstance(music_files, list) else music_files

    if not os.path.exists(music_file):
        print(f"[MP3] File does not exist: {music_file}")
        return

    try:
        pygame.mixer.music.stop()
        pygame.mixer.music.load(music_file)
        pygame.mixer.music.play()
        print(f"[MP3] Playing {emotion_label} - {os.path.basename(music_file)}")
    except Exception as e:
        print(f"[MP3 ERROR] {e}")

# ========== 6. Real-Time Emotion Detection ==========
print("\n[INFO] Launching real-time emotion detection...\n")

emotion_labels = list(train_generator.class_indices.keys())
model = load_model("emotion_model.keras")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

current_emotion = None
last_trigger = time.time() - 10
cooldown = 5  # seconds between emotion-based song changes

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(np.expand_dims(roi, axis=0), axis=-1)

        emotion_label = emotion_labels[np.argmax(model.predict(roi))]

        if emotion_label != current_emotion and time.time() - last_trigger > cooldown:
            current_emotion = emotion_label
            last_trigger = time.time()
            play_emotion_music(emotion_label)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection + Local MP3", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
