from flask import Flask, render_template, Response, request, redirect, url_for
import ctypes
import cv2
import subprocess
import numpy as np
# clib = ctypes.CDLL('C:/Users/sufya/Desktop/smoking detetction/NEW/cv.so')

# clib.d()

app = Flask(__name__)



# @app.route('/')
# def index():
#     return clib.d()
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['GET', 'POST'])
def after():
    img = request.files ['file1']
    img.save('C:/Users/sufya/Desktop/smoking detetction/NEW/static/har-bef.jpg')

    img1 = cv2.imread('C:/Users/sufya/Desktop/smoking detetction/NEW/static/har-bef.jpg')
    cig_cascade = cv2.CascadeClassifier('C:/Users/sufya/Desktop/smoking detetction/NEW/static/naya.xml')
    width = 800
    height = 600
    image = cv2.resize(img1, (width, height))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = cig_cascade.detectMultiScale(gray, 1.02, 13)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(img,'smoking',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),5)
    
    cv2.imwrite('C:/Users/sufya/Desktop/smoking detetction/NEW/static/har-aft.jpg',img)

    return render_template('Faiz.html')



@app.route('/har_vd',methods=['GET','POST'])

def har_vd():

    cap = cv2.VideoCapture(0)
    cig_cascade = cv2.CascadeClassifier('C:/Users/sufya/Desktop/smoking detetction/NEW/static/naya.xml')

    #img = cv2.imread('56.jpg')
    #while cap.isOpened():
    #_, img=cap.read()
    while(True):
        ret,frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = cig_cascade.detectMultiScale(gray, 1.34, 10)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(img,'smoking',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),5)
        #     def stop_streaming(self):
        #         self.stream.release()
        #         self.stopped = True
        #         cv2.destroyAllWindows()
        #         return self.frame

        # img=request.stop_streaming('stp')
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        #cv2.imshow('frame',frame)
    #     if cv2.waitKey(1) & 0xFF == ('q'):
    #         break 

        #return render_template('Faiz.html')

@app.route('/vd_feed')
def vd_feed():
    return Response(har_vd(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/har-web')
def har_web():
    return render_template('har-web.html')

@app.route('/yolo_after', methods=['GET', 'POST'])
def yolo_after():
    img = request.files ['file2']
    img.save('C:/Users/sufya/Desktop/smoking detetction/NEW/static/yolo-bef.jpg')

    net = cv2.dnn.readNet('C:/Users/sufya/Desktop/smoking detetction/NEW/strong.onnx')
        # step 2 - feed a 640x640 image to get predictions

    def format_yolov5(frame):

        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

    image = cv2.imread('C:/Users/sufya/Desktop/smoking detetction/NEW/static/yolo-bef.jpg')
    input_image = format_yolov5(image) # making the image square
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
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

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

        cv2.rectangle(image, box, (0, 255, 255), 2)
        cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1)
        cv2.putText(image, class_list[class_id], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))


    cv2.imwrite('C:/Users/sufya/Desktop/smoking detetction/NEW/static/yolo-aft.jpg',image)

    return render_template('index.html')

# @app.route('/opencam', methods=['GET'])
# def opencam():
#     print("here")
#     subprocess.run(['python3', 'detect.py', '--source', '0'])
#     return render_template('yolo.html')


@app.route('/yolo_vd')

def yolo_vd():
        
    cap = cv2.VideoCapture(0) #1 for bakc camera

    cig_cascade = cv2.CascadeClassifier('C:/Users/sufya/Desktop/smoking detetction/NEW/static/naya.xml')
    net = cv2.dnn.readNet('C:/Users/sufya/Desktop/smoking detetction/NEW/strong.onnx')
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
        ret,image = cap.read()
        input_image = format_yolov5(image) # making the image square
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
            if confidence >= 0.4:

                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if (classes_scores[class_id] > .25):

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

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.4) 

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
            confi= str(result_confidences[i]+0.4)

            cv2.rectangle(image, box, (0, 255, 255), 2)
            cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1)
            cv2.putText(image, class_list[class_id]+" "+ confi, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))

        ret, buffer = cv2.imencode('.jpg', image)
        image = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')


@app.route('/yvd_feed')
def yvd_feed():
    return Response(yolo_vd(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/yolo-web')
def yolo_web():
    return render_template('yolo-web.html')

        

@app.route('/Mujahid')
def mujahid():
    return render_template('Mujahid.html')

@app.route('/Kaif')
def kaif():
    return render_template('Kaif.html')


@app.route('/Faiz')
def faiz():
    return render_template('Faiz.html')

@app.route('/Anas')
def anas():
    return render_template('Anas.html')


@app.route('/open_webcam', methods=['GET'])
def open_webcam():
    print("here")
    subprocess.run(['python3', 'detect.py', '--source', '0'])
    return render_template('index.html')

if __name__ == '__main__':
    app.run(threaded=True, debug=True)





