//
// Created by hw on 5/24/19.
//

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <dirent.h>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

#define MAX_SLIDER_VALUE 255
#define NUM_EIGEN_FACES 10


// Weights for the different eigenvectors
int sliderValues[NUM_EIGEN_FACES];

// Matrices for average (mean) and eigenvectors
Mat averageFace;
vector<Mat> eigenFaces;

Mat createDataMatrix(vector<Mat> &images){
    cout<<"creating data matrix"<<endl;

    Mat ret = Mat(images.size(),images[0].rows*images[0].cols*3,CV_32F);

    for(int k=0;k<images.size();k++){
        Mat img = images[k].reshape(1,1);
        img.copyTo(ret.row(k));
    }
    return ret;
}

void readImages(string path,vector<Mat> &images){
    cout<<"reading images from "<<path.c_str()<<endl;
    DIR *dir;
    struct dirent *pd;
    if((dir=opendir(path.c_str()))){
        if(path.back()!='/')
            path = path + "/";
        while((pd=readdir(dir))){
            if(strcmp(pd->d_name,".")==0||strcmp(pd->d_name,"..")==0)
                continue;
            string fname = pd->d_name;
            Mat img = imread(path+fname);
            img.convertTo(img,CV_32FC3,1/255.);
            images.push_back(img);
            Mat img_flip;
            flip(img,img_flip,1);
            images.push_back(img_flip);
        }
        closedir(dir);
    }
}

void createNewFace(int ,void *){
    Mat output = averageFace.clone();
    for (int i = 0; i <NUM_EIGEN_FACES ; ++i) {
        double weight = sliderValues[i] - MAX_SLIDER_VALUE/2;
        output = output + eigenFaces[i] * weight;
    }
    resize(output,output,Size(),2,2);
    imshow("Result",output);
}

void resetSliderValues(int event,int x,int y,int flags,void *userdata){
    if(event==EVENT_LBUTTONDOWN){
        for(int i=0;i<NUM_EIGEN_FACES;i++){
            sliderValues[i] = 128;
            setTrackbarPos("weights"+to_string(i),"Trackbars",MAX_SLIDER_VALUE/2);
        }
        createNewFace(0,0);
    }
}

int main(int argc,char **argv){
    string path = "../images";
    vector<Mat> images;
    readImages(path,images);

    Size sz = images[0].size();
    Mat data = createDataMatrix(images);
    PCA pca(data,Mat(),PCA::DATA_AS_ROW,NUM_EIGEN_FACES);
    averageFace = pca.mean.reshape(3,sz.height);
    Mat eigenVectors = pca.eigenvectors;
    for (int i = 0; i <NUM_EIGEN_FACES ; ++i) {
        Mat eigenface = eigenVectors.row(i).reshape(3,sz.height);
        eigenFaces.push_back(eigenface);
    }

    Mat output;
    resize(averageFace, output, Size(), 2, 2);
    namedWindow("Result", WINDOW_AUTOSIZE);
    imshow("Result", output);

    // Create trackbars
    namedWindow("Trackbars", WINDOW_AUTOSIZE);
    for (int i = 0; i <NUM_EIGEN_FACES ; ++i) {
        sliderValues[i] = MAX_SLIDER_VALUE/2;
        createTrackbar( "Weight" + to_string(i), "Trackbars", &sliderValues[i], MAX_SLIDER_VALUE, createNewFace);
    }
    setMouseCallback("Result",resetSliderValues);

    waitKey(0);
    destroyAllWindows();

}