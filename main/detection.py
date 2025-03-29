import cv2
def Camera():
   cam = cv2.VideoCapture(0)
   cam.set(3, 740)
   cam.set(4, 580)
   classNames = []
   classFile = 'coco.names'
   with open(classFile, 'rt') as f:
      classNames = f.read().rstrip('\n').split('\n')
   configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
   weightpath = 'frozen_inference_graph.pb'
   net = cv2.dnn_DetectionModel(weightpath, configPath)
   net.setInputSize(320 , 230)
   net.setInputScale(1.0 / 127.5)
   net.setInputMean((127.5, 127.5, 127.5))
   net.setInputSwapRB(True)
   detected_items = []  
   success, img = cam.read()
   classIds, confs, bbox = net.detect(img, confThreshold=0.5)
   if len(classIds) != 0:
      for  classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
         class_name = classNames[classId - 1] 
         if class_name not in detected_items:  
            detected_items.append(class_name)
         cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
         cv2.putText(img, class_name, (box[0] + 10, box[1] + 20), 
            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
   text = ' and '.join(detected_items)
   return text   
Camera()