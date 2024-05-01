import PySimpleGUI as sg
import cv2

layout = [
    [sg.Image(key = '-IMAGE-')],
    [sg.Text('People in image: 0     Eyes in image: 0', key='-TEXT-', expand_x=True, justification='c')]

]
window = sg.Window('Face Detector', layout)
video = cv2.VideoCapture(0)
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml ')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
while True:
    event, values = window.read(timeout=0)
    if event == sg.WIN_CLOSED:
        break

    _, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7, minSize=(50,50))
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7, minSize=(50,50))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    image_bytes = cv2.imencode('.png', frame)[1].tobytes()
    window['-IMAGE-'].update(data=image_bytes)

    window['-TEXT-'].update(f'People in image: {len(faces)}                   Eyes in image : {len(eyes)}')

window.close()

#credits to @jansh7784 - github 
