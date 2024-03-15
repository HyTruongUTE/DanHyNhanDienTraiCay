#suon lap trinh giao dien
import tkinter as tk
from tkinter.filedialog import Open
from PIL import Image, ImageTk
import os
import cv2
import numpy as np

from common import *
from tf_text_graph_common import readTextMessage
from tf_text_graph_ssd import createSSDGraph
from tf_text_graph_faster_rcnn import createFasterRCNNGraph


model = 'yolov8n_trai_cay.onnx'
filename_classes = 'object_detection_classes_yolo.txt'
mywidth  = 640
myheight = 640
postprocessing = 'yolov8'
background_label_id = -1
backend = 0
target = 0

# Load names of classes
classes = None
if filename_classes:
    with open(filename_classes, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

# Load a network
net = cv.dnn.readNet(model)
net.setPreferableBackend(0)
net.setPreferableTarget(0)
outNames = net.getUnconnectedOutLayersNames()

confThreshold = 0.5
nmsThreshold = 0.4
scale = 0.00392
mean = [0, 0, 0]

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    def drawPred(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

        label = '%.2f' % conf

        # Print a label of class.
        if classes:
            assert(classId < len(classes))
            label = '%s: %s' % (classes[classId], label)

        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    layerNames = net.getLayerNames()
    lastLayerId = net.getLayerId(layerNames[-1])
    lastLayer = net.getLayer(lastLayerId)

    classIds = []
    confidences = []
    boxes = []
    if lastLayer.type == 'Region' or postprocessing == 'yolov8':
        # Network produces output blob with a shape NxC where N is a number of
        # detected objects and C is a number of classes + 4 where the first 4
        # numbers are [center_x, center_y, width, height]
        if postprocessing == 'yolov8':
            box_scale_w = frameWidth / mywidth
            box_scale_h = frameHeight / myheight
        else:
            box_scale_w = frameWidth
            box_scale_h = frameHeight

        for out in outs:
            if postprocessing == 'yolov8':
                out = out[0].transpose(1, 0)

            for detection in out:
                scores = detection[4:]
                if background_label_id >= 0:
                    scores = np.delete(scores, background_label_id)
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * box_scale_w)
                    center_y = int(detection[1] * box_scale_h)
                    width = int(detection[2] * box_scale_w)
                    height = int(detection[3] * box_scale_h)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    else:
        print('Unknown output layer type: ' + lastLayer.type)
        exit()

    # NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
    # or NMS is required if number of outputs > 1
    if len(outNames) > 1 or (lastLayer.type == 'Region' or postprocessing == 'yolov8') and 0 != cv.dnn.DNN_BACKEND_OPENCV:
        indices = []
        classIds = np.array(classIds)
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        unique_classes = set(classIds)
        for cl in unique_classes:
            class_indices = np.where(classIds == cl)[0]
            conf = confidences[class_indices]
            box  = boxes[class_indices].tolist()
            nms_indices = cv.dnn.NMSBoxes(box, conf, confThreshold, nmsThreshold)
            indices.extend(class_indices[nms_indices])
    else:
        indices = np.arange(0, len(classIds))

    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.filename = None
        self.title('nhận dạng vật thể')
        self.result_label = tk.Label(self, text="", font=("Arial", 16, "bold"), fg="black")
        self.result_label.grid(row=2, column=0, padx=5, pady=5)
        self.cvs_image = tk.Canvas(self, width=600, height=600, 
                              relief= tk.SUNKEN, border= 1)
        
        lbl_frm_menu = tk.LabelFrame(self)
        btn_open_image = tk.Button(lbl_frm_menu, text = 'Open Image', width = 9, 
                                   command=self.btn_open_image_click)

        btn_predict_image = tk.Button(lbl_frm_menu, text = 'Predict', width=9,
                                      command=self.btn_predict_image_click)
        
        btn_open_image.grid(row=0, column=0, padx=5, pady=5)
        btn_predict_image.grid(row=1, column=0, padx=5, pady=5)

        self.cvs_image.grid(row = 0, column= 0, padx=5, pady=5)
        lbl_frm_menu.grid(row=0, column=1,padx=5, pady=5, sticky=tk.N)

    def btn_open_image_click(self):
        ftypes = [('Image', '*.jpg *.tif *.bmp *.gif *.png')]
        dlg = Open(self, filetypes = ftypes)
        self.filename = dlg.show()
        if self.filename !='':
            image = Image.open(self.filename)
            self.image_tk = ImageTk.PhotoImage(image)
            self.cvs_image.create_image(0,0,anchor = tk.NW, image =self.image_tk)

    def phan_nguong(self,imgin):
        M, N, C = imgin.shape
        imgout = np.zeros((M,N), np.uint8)
        mau_nen = imgin[0,0]
        for x in range(0, M):
            for y in range(0,N):
                r = imgin[x,y]
                if r[0]==mau_nen[0] and mau_nen[1] == r[1] and r[2] == mau_nen[2]:
                    imgout[x,y] =0
                else:
                    imgout[x,y] = 255
        return imgout

    def btn_predict_image_click(self):
        frame = cv2.imread(self.filename, cv2.IMREAD_COLOR)
        if not frame is None:
            frameHeight = frame.shape[0]
            frameWidth = frame.shape[1]
            # Create a 4D blob from a frame.
            inpWidth = mywidth if mywidth else frameWidth
            inpHeight = myheight if myheight else frameHeight
            blob = cv.dnn.blobFromImage(frame, size=(inpWidth, inpHeight), swapRB=True, ddepth=cv.CV_8U)
            # Run a model
            net.setInput(blob, scalefactor=scale, mean=mean)
            if net.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
                frame = cv.resize(frame, (inpWidth, inpHeight))
                net.setInput(np.array([[inpHeight, inpWidth, 1.6]], dtype=np.float32), 'im_info')

            outs = net.forward(outNames)
            postprocess(frame, outs)     
            color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_imge = Image.fromarray(color_coverted)
            self.image_tk = ImageTk.PhotoImage(pil_imge)
            self.cvs_image.create_image(0,0,anchor = tk.NW, image =self.image_tk)

if __name__  == "__main__":
    app =  App()
    app.mainloop()