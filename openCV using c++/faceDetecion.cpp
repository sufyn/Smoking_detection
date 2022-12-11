#include<iostream>
#include<Opencv2/imgcodecs.hpp>
#include<opencv/imgproc.hpp>
#include<opencv/highgui.hpp>
#include<opencv/objdetect.hpp>

using namespace cv;
using namespace std;

void main(){
    string path = "Resources/test.png";
    mat img = imread(path);

    CascadecClassifier faceCascade;
    faceCascade.load("Resoureces/haarcascade_frontalface_default.xml");

    if (faceCascade.empty()) {
        cout<< "XML file not loaded"<<  endln;
    }
    vector<Rect> faces;
    faceCascade.detectMultiScale(img,faces,1.1,10);

    for(int i=0; i<faces.size();i++){
        rectangle(img,faces[i].t1()faces[i].br(),Scalar(225,0,255,3));
    }
}