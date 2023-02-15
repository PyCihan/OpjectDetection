import os
import django
from Detector import *
import RPi.GPIO as GPIO
import time
import os

time.sleep(30)
SENSOR_PIN = 23
GPIO.setmode(GPIO.BCM)
GPIO.setup(SENSOR_PIN, GPIO.IN)

counter = 1

def image_eval(channel):
    global counter
    imgName = "img" + str(counter)
    os.system('libcamera-still -o /home/pi/Desktop/Project/https---github.com-PyCihan-Projekt/OpjectDetection/media/images' + imgName + ' -t 1')
    time.sleep(3)

    os.environ.setdefault( 'DJANGO_SETTINGS_MODULE', 'OpjectDetection.settings')
    django.setup()
    from ImageServer.models import Image

    imagePath = 'media/images/' + str(imgName) + '.jpeg'

    modelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'

    classFile = 'coco.names'
    detector = Detector()
    detector.readClasses(classFile)
    detector.downloadModel(modelURL)
    detector.loadModel()
    detector.predictImage(imagePath)

    image = Image()
    image.image = imagePath
    image.label = Detector.keys
    image.save()

    counter += 1
    print('Es gab eine Bewegung!')
    print('Neuer Counter: ' + str(counter))
    
try:
    GPIO.add_event_detect(SENSOR_PIN , GPIO.RISING, callback=image_eval)
    while True:
      time.sleep(100)
except KeyboardInterrupt:
    print("Beende...")
GPIO.cleanup()




