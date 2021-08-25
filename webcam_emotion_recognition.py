import cv2
import numpy as np
import tensorflow as tf
import os
from zipfile import ZipFile
from google_drive_downloader import GoogleDriveDownloader as gdd


def load_model():
    if '3_trt' in os.listdir('./'):
        emotion_model = tf.keras.models.load_model('3_trt/')
    elif 'model.zip' in os.listdir('./'):
        with ZipFile('model.zip', 'r') as zipObj:
            ZipFile.extractall(zipObj)
        emotion_model = tf.keras.models.load_model('3_trt/')
    else:
        gdd.download_file_from_google_drive('1jkwvE0XvX919wYkD4ixEk6VYjdQFiLIx', './model.zip')
        with ZipFile('model.zip', 'r') as zipObj:
            ZipFile.extractall(zipObj)

    emotion_model = tf.keras.models.load_model('3_trt/')
    print('model downloaded')

    return emotion_model


def load_emotion_dict():
    emo_dict = {0: 'anger',
                1: 'contempt',
                2: 'disgust',
                3: 'fear',
                4: 'happy',
                5: 'neutral',
                6: 'sad',
                7: 'surprise',
                8: 'uncertain'}
    return emo_dict


def decode_prediction(pred: np.ndarray, emo_dict):
    return np.array([emo_dict[x] for x in pred])


# def log_error(func):
#
#     def inner(*args, **kwargs):
#         try:
#             return func(*args, **kwargs)
#         except Exception as e:
#             print(f'Ошибка: {e}')
#
#     return inner
#
#
# @log_error
def main():
    print(tf.config.list_physical_devices('GPU'))

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Не удалось открыть камеру")
    else:
        print("Камера запущена")
    emotion_dict = load_emotion_dict()
    emotion_model = load_model()
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # здесь мы в бесконечном цикле считываем кадр с камеры и выводим его, используя cv2.imshow()
    # корректная остановка окна с камерой произойдет, когда мы нажмем q на клавиатуре
    while (True):
        try:
            ret, frame = cam.read()
            grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(grayscale_frame, 1.3, 5)  # or 1.1, 19
            if len(faces) > 0:
                faces_collection = []
                bb_collection = []

                for (x, y, w, h) in faces:
                    face_boundingbox_bgr = frame[y:y + h, x:x + w]
                    face_boundingbox_rgb = cv2.cvtColor(face_boundingbox_bgr, cv2.COLOR_BGR2RGB)
                    faces_collection.append(face_boundingbox_rgb)
                    bb_collection.append((x, y, w, h))

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), )

                faces_collection = np.array(faces_collection)
                faces_collection = tf.image.resize(faces_collection, (150, 150))
                preds = emotion_model(faces_collection)

                # preds -> emotion_idx, confidence_percentage
                emotion_idx = np.argmax(preds, axis=1)
                emotions = decode_prediction(emotion_idx, emotion_dict)
                confidence_percentage = np.max(preds, axis=1)

                for bb, emotion, confidence in zip(bb_collection, emotions, confidence_percentage):
                    cv2.putText(frame, f'{emotion:9s} {confidence:.0%}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (30, 255, 30), 1)

            cv2.imshow("facial emotion recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f'Ошибка: {e}')

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

