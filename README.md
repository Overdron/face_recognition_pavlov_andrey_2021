# face_recognition_pavlov_andrey_2021
В данном репозитории представлены обучение и инференс моделей по распознаванию эмоций, и также приложение для веб камеры.

[EmotionRecognition_PavlovAndrey_2021_training.ipynb](https://github.com/Overdron/face_recognition_pavlov_andrey_2021/blob/main/EmotionRecognition_PavlovAndrey_2021_training.ipynb) - Полный пайплайн обучения модели по распознаванию эмоций человека, включая valence-arousal модель. Запускать тетрадку заново не имеет смысла, она выгружена просто для ознакомления. Протестировать модель можно в следующей тетрадке. Выполнено в googlecolab. Сохранение и загрузка моделей ссылается на googledrive.

[EmotionRecognition_PavlovAndrey_2021_inference](https://github.com/Overdron/face_recognition_pavlov_andrey_2021/blob/main/EmotionRecognition_PavlovAndrey_2021_inference.ipynb) - Загрузка и инференс предобученных моделей распознавания эмоций. По сути это та же первая тетрадка, из которой было вырезано обучение. Для запуска достаточно выполнить все ячейки. Выполнено в googlecolab.

[webcam_emotion_recognition.py](https://github.com/Overdron/face_recognition_pavlov_andrey_2021/blob/main/webcam_emotion_recognition.py) - приложение подключается к вебкамере и опознает эмоцию, используя модель, обученную в перевой тетрадке.

Ссылки, по которым доступны обученные модели:
- модель для определения категории эмоции - https://drive.google.com/drive/folders/1AHaRMYHqJStvEuXV02lWHtXU_rmNFSWv?usp=sharing
- модель для определения значений valence-arousal эмоции - https://drive.google.com/drive/folders/1S0CZ2JFHO8up6RVdAeQqdIaKbl8gnBGM?usp=sharing
