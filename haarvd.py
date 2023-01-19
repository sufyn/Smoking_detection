import  cv2
import  numpy  as  np
import winsound
# cap = cv2.VideoCapture('C:/Users/sufya/Desktop/smoking detetction/NEW/Smoking_detection.mp4') #1 for bakc camera
cap = cv2.VideoCapture(1) #1 for bakc camera

net = cv2.dnn.readNet('C:/Users/sufya/Desktop/smoking detetction/NEW/nm.onnx')
# net = cv2.dnn.readNet('C:/Users/sufya/Desktop/smoking detetction/NEW/mlk.onnx')
            # step 2 - feed a 640x640 image to get predictions
def format_yolov5(frame):

            row, col, _ = frame.shape
            _max = max(col, row)
            result = np.zeros((_max, _max, 3), np.uint8)
            result[0:row, 0:col] = frame
            return result


    #img = cv2.imread('56.jpg')
    #while cap.isOpened():
    #_, img=cap.read()
while(True):
        ret,frame = cap.read()
        input_image = format_yolov5(frame) # making the image square
        blob = cv2.dnn.blobFromImage(input_image , 1/255.0, (640, 640), swapRB=True)
        net.setInput(blob)
        predictions = net.forward()

        # step 3 - unwrap the predictions to get the object detections 

        
        class_ids = []
        confidences = []
        boxes = []

        output_data = predictions[0]

        image_width, image_height, _ = input_image.shape
        x_factor = image_width / 640
        y_factor =  image_height / 640

        for r in range(25200):
            row = output_data[r]
            confidence = row[4]
            if confidence >= 0.18:

                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if (classes_scores[class_id] > .18):

                    confidences.append(confidence)

                    class_ids.append(class_id)

                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        class_list = ['smoking']

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.18, 0.18) 

        result_class_ids = []
        result_confidences = []
        result_boxes = []

        for i in indexes:
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])

        for i in range(len(result_class_ids)):

            box = result_boxes[i]
            class_id = result_class_ids[i]
            # confi= str(result_confidences[i]+0.4)
            confi= str(100*result_confidences[i].round(2)+40)+"%"

            cv2.rectangle(frame, box, (0, 255, 255), 2)
            cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1)
            cv2.putText(frame, class_list[class_id]+" "+ confi, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))
            winsound.Beep(500,200)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   
cap.release()
cv2.destroyAllWindows()

