import cv2
from playsound import playsound
import speech_recognition as sr
import RPi.GPIO as gpio
import time

def init():
 gpio.setmode(gpio.BCM)
 gpio.setup(17, gpio.OUT)
 gpio.setup(23, gpio.OUT)
 gpio.setup(24, gpio.OUT)
 gpio.setup(25, gpio.OUT)
def forward(sec):
 init()
 gpio.output(17, True)
 gpio.output(23, False)
 gpio.output(24, True)
 gpio.output(25, False)
 time.sleep(sec)
 gpio.cleanup()
def reverse(sec):
 init()
 gpio.output(17, False)
 gpio.output(23, True)
 gpio.output(24, False)
 gpio.output(25, True)
 time.sleep(sec)
 gpio.cleanup()

for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))

r = sr.Recognizer()
m = sr.Microphone()

try:
    print("A moment of silence, please...")
    with m as source: r.adjust_for_ambient_noise(source)
    print("Set minimum energy threshold to {}".format(r.energy_threshold))
    while True:
        print("Say something!")
        with m as source: audio = r.listen(source)
        print("Got it! wait...")
        try:
            # recognize speech using Google Speech Recognition
            value = r.recognize_google(audio)
            print("You said {}".format(value))
            if value != 'Michael':
                playsound('/home/criuser/Téléchargements/OD_Robot/sup.wav')

            if value == 'Michael': #This is the keyword , if recognised , then it will start working
                playsound('/home/criuser/Téléchargements/OD_Robot/surprise-motherfucker.wav')


                classNames = []
                classFile = "/home/criuser/Téléchargements/OD_Robot/coco.names"
                with open(classFile, "rt") as f:
                    classNames = f.read().rstrip("\n").split("\n")

                configPath = "/home/criuser/Téléchargements/OD_Robot/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
                weightsPath = "/home/criuser/Téléchargements/OD_Robot/frozen_inference_graph.pb"

                net = cv2.dnn_DetectionModel(weightsPath, configPath)
                net.setInputSize(320, 320)
                net.setInputScale(1.0 / 127.5)
                net.setInputMean((127.5, 127.5, 127.5))
                net.setInputSwapRB(True)


                def getObjects(img, thres, nms, draw=True, objects=[]):
                    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)

                    if len(objects) == 0: objects = classNames
                    objectInfo = []
                    if len(classIds) != 0:
                        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                            className = classNames[classId - 1]
                            if className in objects:
                                objectInfo.append([box, className])
                                if (draw):
                                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)

                                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                    return img, objectInfo


                if __name__ == "__main__":

                    cap = cv2.VideoCapture(0)
                    cap.set(3, 640)
                    cap.set(4, 480)

                    while True:
                        success, img = cap.read()
                        result, objectInfo = getObjects(img, 0.45, 0.2,objects=['cup','person'])
                        cv2.imshow("Output", img)
                        cv2.waitKey(10)
                        if    (objectInfo !=[]):
                            if objectInfo[0][1] == 'person':
                                    playsound('/home/criuser/Téléchargements/OD_Robot/surprise-motherfucker.wav')
                                    forward(5)
                            if objectInfo[0][1] == 'cup':
                                    playsound('/home/criuser/Téléchargements/OD_Robot/vodka.wav')
                                    reverse(5)


        except sr.UnknownValueError:
            print("Oops! Raise Your Voice Pal")
            playsound('/home/criuser/Téléchargements/OD_Robot/congratulations.wav')
        except sr.RequestError as e:
            print("Oops! Couldn't request results from Google Speech Recognition service; {0}".format(e))
except KeyboardInterrupt:
    pass

