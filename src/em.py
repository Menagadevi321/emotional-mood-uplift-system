"""
Mood Uplift System
------------------
- First run : trains the model, tests it, saves the model + labels.
- Next runs : loads the saved model and starts the webcam straight away.
- To retrain: python emo.py --retrain
"""

import os
import sys
import json
import time
import random
import argparse

import numpy as np
import cv2
import pygame  # local mp3 playback

# ========== Parameters ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "emotion_labels.json")
PLOT_PATH = os.path.join(BASE_DIR, "training_accuracy.png")

IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 50

TRAIN_DIR = r"D:/emotion dataset 4/images/train"
TEST_DIR = r"D:/Emotion dataset 1/test"

MUSIC_DIR = r"C:/Users/AUSU/Desktop/songs"

# Used only if emotion_labels.json is missing (order = alphabetical folder names)
DEFAULT_LABELS = ["angry", "fearful", "happy", "neutral", "sad"]

COOLDOWN = 5  # seconds between emotion-based song changes


# ========== Training (runs only when needed) ==========
def train_and_save():
    # TensorFlow is imported here so it is only needed when training
    import matplotlib
    matplotlib.use("Agg")  # save plots to a file, don't block with a window
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
    )
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    if not os.path.isdir(TRAIN_DIR):
        sys.exit(
            f"[ERROR] Training folder not found:\n  {TRAIN_DIR}\n"
            "Fix TRAIN_DIR at the top of emo.py, or put a trained "
            f"'{os.path.basename(MODEL_PATH)}' next to this script."
        )

    print("[INFO] Preparing data...")
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode="categorical", subset="training", color_mode="grayscale",
    )
    val_gen = datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode="categorical", subset="validation", color_mode="grayscale",
    )

    # Save labels in the right order so we never need the dataset again
    labels = [name for name, _ in sorted(
        train_gen.class_indices.items(), key=lambda kv: kv[1])]

    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(128, activation="relu"),
        Dense(train_gen.num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])

    print("[INFO] Training model (this only happens once)...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        # stops early when validation stops improving -> fewer wasted epochs
        callbacks=[EarlyStopping(monitor="val_accuracy", patience=5,
                                 restore_best_weights=True)],
    )

    # Save model and labels
    model.save(MODEL_PATH)
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f)
    print(f"[INFO] Saved model  -> {MODEL_PATH}")
    print(f"[INFO] Saved labels -> {LABELS_PATH}  {labels}")

    # Save accuracy plot as an image (no blocking window)
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"[INFO] Saved accuracy plot -> {PLOT_PATH}")

    # Evaluate on the test set (if available)
    if os.path.isdir(TEST_DIR):
        test_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
            TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
            class_mode="categorical", shuffle=False, color_mode="grayscale",
        )
        loss, acc = model.evaluate(test_gen, verbose=0)
        print(f"\nTest Loss: {loss:.4f}, Accuracy: {acc:.4f}")

        y_pred = np.argmax(model.predict(test_gen, verbose=0), axis=1)
        print("\nClassification Report:\n")
        print(classification_report(
            test_gen.classes, y_pred,
            target_names=list(test_gen.class_indices.keys())))
        print("Confusion Matrix:")
        print(confusion_matrix(test_gen.classes, y_pred))
    else:
        print("[INFO] Test folder not found, skipping evaluation.")

    return model, labels


# ========== Loading a saved model ==========
def load_saved():
    from tensorflow.keras.models import load_model

    print("[INFO] Loading saved model (no training needed)...")
    model = load_model(MODEL_PATH)

    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            labels = json.load(f)
    else:
        labels = DEFAULT_LABELS
        print("[WARN] emotion_labels.json not found, using default labels:",
              labels)

    n_out = model.output_shape[-1]
    if len(labels) != n_out:
        sys.exit(
            f"[ERROR] Model has {n_out} outputs but there are {len(labels)} "
            "labels. Run: python emo.py --retrain"
        )
    return model, labels


# ========== Music ==========
def song(folder, *names):
    return [f"{MUSIC_DIR}/{folder}/{n}" for n in names]


EMOTION_MUSIC = {
    "angry":   song("angry", "calm1.mp3", "calm2.mp3", "calm3.mp3",
                    "calm4.mp3", "calm5.mp3"),
    "fearful": song("fear", "mt1.mp3", "mt2.mp3", "mt3.mp3",
                    "mt4.mp3", "mt5.mp3"),
    "happy":   song("happy", "h1.mp3", "h2.mp3", "h3.mp3",
                    "h4.mp3", "h5.mp3"),
    "neutral": song("neutral", "song1.mp3", "song2.mp3", "song3.mp3"),
    "sad":     song("sad", "sad1.mp3", "sad2.mp3", "sad3.mp3", "sad4.mp3"),
}


def play_emotion_music(emotion):
    files = EMOTION_MUSIC.get(emotion)
    if not files:
        print(f"[MP3] No music configured for emotion: {emotion}")
        return

    music_file = random.choice(files)
    if not os.path.exists(music_file):
        print(f"[MP3] File does not exist: {music_file}")
        return

    try:
        pygame.mixer.music.stop()
        pygame.mixer.music.load(music_file)
        pygame.mixer.music.play()
        print(f"[MP3] Playing {emotion} - {os.path.basename(music_file)}")
    except Exception as e:
        print(f"[MP3 ERROR] {e}")


# ========== Real-time detection ==========
def run_webcam(model, labels):
    pygame.mixer.init()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)  # try 1 if your webcam is not found

    if not cap.isOpened():
        sys.exit("[ERROR] Could not open webcam.")

    current_emotion = None
    last_trigger = time.time() - 10

    print("[INFO] Webcam started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(np.expand_dims(roi, axis=0), axis=-1)

            preds = model.predict(roi, verbose=0)
            emotion = labels[int(np.argmax(preds))]

            if emotion != current_emotion and time.time() - last_trigger > COOLDOWN:
                current_emotion = emotion
                last_trigger = time.time()
                play_emotion_music(emotion)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Mood Uplift System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()


# ========== Main ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true",
                        help="ignore the saved model and train again")
    args = parser.parse_args()

    if os.path.exists(MODEL_PATH) and not args.retrain:
        model, labels = load_saved()
    else:
        model, labels = train_and_save()

    run_webcam(model, labels)