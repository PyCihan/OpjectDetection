import cv2, time, os, tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(123)

class Detector:
    keys = []
    
    def __init__(self):
        pass
    
    def readClasses(self, classesFilePath): #liest die Klassen aus der Datei ein
        with open(classesFilePath, 'r') as f: 
            self.classesList = f.read().splitlines() #splitlines() trennt die Zeilen der Datei in eine Liste
            
            
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3)) #erstellt eine Liste mit zufälligen Farben
        
        #print((len(self.classesList), len(self.colorList)))
        
    def downloadModel(self, modelURL):  #lädt das Modell aus dem Internet herunter
        fileName = os.path.basename(modelURL)  #os.path.basename() gibt den Dateinamen zurück
        self.modelName = fileName[:fileName.index('.')] 
        
        #print(fileName)
        #print(self.modelName)
        
        self.cacheDir = "./pretrained_models"
        os.makedirs(self.cacheDir, exist_ok=True)
        
        get_file(fname = fileName, 
                 origin = modelURL, 
                 cache_dir = self.cacheDir,
                 cache_subdir="checkpoints",
                 extract=True
                 )
        
    def loadModel(self): #lädt das Modell
        print('Loading Model ' + self.modelName)
        tf.keras.backend.clear_session() 
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model")) #os.path.join() verbindet die Pfade zu einem Pfad
        
        print("Model " + self.modelName + "loaded succesfully")
    
    def createBoundingBox(self, image):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB) #wandelt das Bild in das RGB-Format um
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8) #wandelt das Bild in ein Tensor um
        inputTensor = inputTensor[tf.newaxis,...] 
        
        detections = self.model(inputTensor) #gibt die Detektionen zurück
        bboxs = detections['detection_boxes'][0].numpy() 
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()
        
        #print("ClassIndexes:" + classIndexes)
        #print("Detections: 2" + detections)
        #print("ClassScores: " + classScores)
        
        imH, imW, imC = image.shape
        
        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50,
        iou_threshold=0.4, score_threshold=0.2)
        
        #print(bboxIdx)
        
        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist()) #wandelt die Bounding Box in eine Liste um
                classConfidence = round(100*classScores[i]) #rundet die Wahrscheinlichkeit auf 2 Nachkommastellen
                classIndex = classIndexes[i]
                
                classLabelText = self.classesList[classIndex]
                classColor = self.colorList[classIndex]
                
                displayText = '{}: {}%'.format(classLabelText, classConfidence) #gibt die Wahrscheinlichkeit in Prozent aus
                
                ymin, xmin, ymax, xmax = bbox #gibt die Koordinaten der Bounding Box aus
                
                #print(ymin, xmin, ymax, xmax)
                #break
                
                xmin, xmax, ymin, ymax = (xmin*imW, xmax*imW, ymin*imH, ymax*imH)
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
                
                cv2.rectangle(image, (xmin,ymin), (xmax,ymax), color= classColor, thickness=3)
                cv2.putText(image, displayText, (xmin, ymin-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2) #gibt den Text auf dem Bild aus
                
                Detector.keys.append(classLabelText) #fügt die Klassen in die Liste hinzu
                
        return image 

    
    def predictImage(self,imagePath):
        image = cv2.imread(imagePath)
        bboxImage = self.createBoundingBox(image)
        cv2.imwrite(imagePath, bboxImage)
        
        #cv2.imshow("Result", bboxImage)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()