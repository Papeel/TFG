import numpy as np
import cv2
from tensorflow.python.keras.models import load_model
from mtcnn.mtcnn import MTCNN

shape = (100, 100)
name_model = 'v163_3l9.h5'
top = 0
left = 0
add_px = 50
factor = 0.7
cap = cv2.VideoCapture(0)
model = load_model(name_model)
detector = MTCNN()
print()

while True:

    ret, imagen = cap.read()

    dets = detector.detect_faces(imagen)

    real = imagen[:, :, :]
    cuadros = []
    for d in dets:

        left = max(0, int(d['box'][0] - d['box'][2] * factor))
        top = max(0, int(d['box'][1] - d['box'][3] * factor))
        heigh = int(d['box'][3] + d['box'][3] * factor)
        width = int(d['box'][2] + d['box'][2] * factor)

        frame = real[top:d['box'][1] + heigh, left: d['box'][0] + width, :]
        frame = cv2.resize(frame, dsize=shape)
        frame = frame / 255.

        y_pred = model.predict(x=[[frame]])
        y_pred = y_pred[0]
        result = np.argmax(y_pred)

        print(result)
        if result == 0:
            cuadros.append([(left, top),(d['box'][0] + width, d['box'][1] + heigh),(255, 0, 0)])
        elif result == 1:
            cuadros.append([(left, top), (d['box'][0] + width, d['box'][1] + heigh), (0, 255, 0)])
        else:
            cuadros.append([(left, top), (d['box'][0] + width, d['box'][1] + heigh), (0, 0, 255)])


    for c in cuadros:
        cv2.rectangle(imagen, c[0], c[1], c[2])

    cv2.imshow('frame', imagen)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
