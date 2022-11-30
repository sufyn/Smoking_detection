from flask import Flask, render_template, Response, request, redirect, url_for
import ctypes
import cv2


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
    img.save('NEW/static/file.jpg')

    img1 = cv2.imread('NEW/static/file.jpg')
    cig_cascade = cv2.CascadeClassifier('NEW/static/cascade.xml')
    width = 800
    height = 600
    image = cv2.resize(img1, (width, height))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = cig_cascade.detectMultiScale(gray, 1.02, 13)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(img,'smoking',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),5)
    
    cv2.imwrite('NEW/static/after.jpg',img)

    return render_template('after.html')



@app.route('/Mujahid')
def mujahid():
    return render_template('Mujahid.html')

@app.route('/Kaif')
def kaif():
    return render_template('Kaif.html')


if __name__ == '__main__':
    app.run(debug=True)





