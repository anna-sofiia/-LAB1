import os
import cv2
import gzip
import pickle
import string
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


TRAIN_IMAGES_PATH = "emnist-letters-train-images-idx3-ubyte.gz"
TRAIN_LABELS_PATH = "emnist-letters-train-labels-idx1-ubyte.gz"
TEST_IMAGES_PATH = "emnist-letters-test-images-idx3-ubyte.gz"
TEST_LABELS_PATH = "emnist-letters-test-labels-idx1-ubyte.gz"

MODEL_PATH = "emnist_letters_model.h5"
HISTORY_PATH = "training_history.pkl"

IMG_SIZE = 28
NUM_CLASSES = 26



# зчитування EMNIST

def load_emnist_images(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    images = data.reshape(-1, 28, 28)
    return images


def load_emnist_labels(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


# підготовка даних EMNIST

def preprocess_emnist_images(images):
    processed = []

    for img in images:
        img = np.transpose(img)
        img = img.astype("float32") / 255.0
        processed.append(img)

    processed = np.array(processed)
    processed = processed.reshape(-1, 28, 28, 1)
    return processed


def preprocess_labels(labels):
    labels = labels - 1
    return to_categorical(labels, NUM_CLASSES)


def label_to_letter(label_index):
    return string.ascii_lowercase[label_index]


# побудова моделі
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),

        Flatten(),

        Dense(256, activation='relu'),
        Dropout(0.4),

        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# графіки навчання

def plot_training_history_from_dict(history_dict):
    plt.figure(figsize=(8, 5))
    plt.plot(history_dict['loss'], label='Train loss')
    plt.plot(history_dict['val_loss'], label='Validation loss')
    plt.title('Графік функції втрат')
    plt.xlabel('Епоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(history_dict['accuracy'], label='Train accuracy')
    plt.plot(history_dict['val_accuracy'], label='Validation accuracy')
    plt.title('Графік точності')
    plt.xlabel('Епоха')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def save_history(history):
    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history.history, f)


def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "rb") as f:
            return pickle.load(f)
    return None


def show_test_predictions(model, x_test, y_test, count=10):
    predictions = model.predict(x_test[:count], verbose=0)

    plt.figure(figsize=(15, 6))

    for i in range(count):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')

        predicted_class = np.argmax(predictions[i])
        true_class = np.argmax(y_test[i])

        pred_letter = label_to_letter(predicted_class)
        true_letter = label_to_letter(true_class)

        plt.title(f"True: {true_letter}\nPred: {pred_letter}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def show_saved_test_predictions(model, count=10):
    print("Завантаження тестових даних для показу прикладів...")

    x_test = load_emnist_images(TEST_IMAGES_PATH)
    y_test = load_emnist_labels(TEST_LABELS_PATH)

    x_test = preprocess_emnist_images(x_test)
    y_test = preprocess_labels(y_test)

    show_test_predictions(model, x_test, y_test, count=count)

def check_predictions_distribution(model, x_test, y_test, count=1000):
    predictions = model.predict(x_test[:count], verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test[:count], axis=1)

    print("\nПерші 20 справжніх класів:")
    print(true_classes[:20])

    print("\nПерші 20 передбачених класів:")
    print(pred_classes[:20])

    unique, counts = np.unique(pred_classes, return_counts=True)

    print("\nРозподіл передбачених класів:")
    for u, c in zip(unique, counts):
        print(f"{label_to_letter(u)}: {c}")



def choose_image():
    root = Tk()
    root.withdraw()
    root.lift()
    root.focus_force()
    root.attributes('-topmost', True)
    root.update()

    file_path = filedialog.askopenfilename(
        parent=root,
        title="Оберіть зображення з буквою",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
    )

    root.destroy()
    return file_path

def preprocess_uploaded_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Не вдалося завантажити зображення.")

    original = img.copy()

    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        raise ValueError("На зображенні не знайдено букву.")

    x, y, w, h = cv2.boundingRect(coords)
    letter = thresh[y:y + h, x:x + w]

    size = max(w, h) + 20
    square = np.zeros((size, size), dtype=np.uint8)

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = letter

    resized = cv2.resize(square, (20, 20))

    final_img = np.zeros((28, 28), dtype=np.uint8)
    final_img[4:24, 4:24] = resized

    normalized = final_img.astype("float32") / 255.0
    normalized = normalized.reshape(1, 28, 28, 1)

    return original, final_img, normalized



def predict_uploaded_image(model):
    print("Відкриваємо вікно вибору файлу...")
    file_path = choose_image()
    print("Вікно вибору файлу закрите.")

    if not file_path:
        print("Файл не обрано.")
        return

    try:
        print("Обробка зображення...")
        original, processed, prepared = preprocess_uploaded_image(file_path)

        print("Розпізнавання...")
        prediction = model.predict(prepared, verbose=0)
        predicted_class = np.argmax(prediction)
        predicted_letter = label_to_letter(predicted_class)
        confidence = np.max(prediction) * 100

        print(f"\nРозпізнана буква: {predicted_letter}")
        print(f"Ймовірність: {confidence:.2f}%")

        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(original, cmap='gray')
        plt.title("Оригінальне зображення")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(processed, cmap='gray')
        plt.title(f"Після обробки\nПрогноз: {predicted_letter}")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Помилка під час обробки зображення: {e}")


def prepare_data():
    print("Завантаження даних EMNIST...")

    x_train_full = load_emnist_images(TRAIN_IMAGES_PATH)
    y_train_full = load_emnist_labels(TRAIN_LABELS_PATH)

    x_test = load_emnist_images(TEST_IMAGES_PATH)
    y_test = load_emnist_labels(TEST_LABELS_PATH)

    print("Попередня обробка даних...")
    x_train_full = preprocess_emnist_images(x_train_full)
    x_test = preprocess_emnist_images(x_test)

    y_train_full = preprocess_labels(y_train_full)
    y_test = preprocess_labels(y_test)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.15,
        random_state=42
    )

    print(f"Навчальна вибірка: {x_train.shape[0]}")
    print(f"Контрольна вибірка: {x_val.shape[0]}")
    print(f"Тестова вибірка: {x_test.shape[0]}")

    return x_train, x_val, y_train, y_val, x_test, y_test


def train_and_save_model():
    x_train, x_val, y_train, y_val, x_test, y_test = prepare_data()

    model = build_model()

    print("Навчання моделі...")
    history = model.fit(
        x_train,
        y_train,
        epochs=15,
        batch_size=128,
        validation_data=(x_val, y_val),
        verbose=1
    )

    print("Збереження моделі...")
    model.save(MODEL_PATH)
    save_history(history)

    print("Оцінка на тестовій вибірці...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Точність на тестовій вибірці: {test_acc:.4f}")
    print(f"Функція втрат на тестовій вибірці: {test_loss:.4f}")

    plot_training_history_from_dict(history.history)
    show_test_predictions(model, x_test, y_test, count=10)
    check_predictions_distribution(model, x_test, y_test, count=1000)

    return model


def main():
    history_data = load_history()

    if os.path.exists(MODEL_PATH):
        print("Знайдено збережену модель. Завантаження...")
        model = load_model(MODEL_PATH)
        print("Модель успішно завантажена.")
    else:
        model = train_and_save_model()
        history_data = load_history()

    while True:
        print("\nМеню:")
        print("1 - Розпізнати власне зображення з буквою")
        print("2 - Показати графіки навчання")
        print("3 - Показати приклади передбачень (True / Pred)")
        print("4 - Перенавчити модель")
        print("5 - Вийти")

        choice = input("Оберіть дію: ")

        if choice == "1":
            predict_uploaded_image(model)

        elif choice == "2":
            history_data = load_history()
            if history_data is not None:
                plot_training_history_from_dict(history_data)
            else:
                print("Історія навчання не знайдена. Спочатку потрібно навчити модель.")

        elif choice == "3":
            show_saved_test_predictions(model, count=10)

        elif choice == "4":
            print("Перенавчання моделі...")
            model = train_and_save_model()
            history_data = load_history()

        elif choice == "5":
            print("Програма завершена.")
            break

        else:
            print("Неправильний вибір. Спробуйте ще раз.")


if __name__ == "__main__":
    main()