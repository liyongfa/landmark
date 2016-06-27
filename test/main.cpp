// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
 
 This example program shows how to find frontal human faces in an image and
 estimate their pose.  The pose takes the form of 68 landmarks.  These are
 points on the face such as the corners of the mouth, along the eyebrows, on
 the eyes, and so forth.
 
 
 
 This face detector is made using the classic Histogram of Oriented
 Gradients (HOG) feature combined with a linear classifier, an image pyramid,
 and sliding window detection scheme.  The pose estimator was created by
 using dlib's implementation of the paper:
 One Millisecond Face Alignment with an Ensemble of Regression Trees by
 Vahid Kazemi and Josephine Sullivan, CVPR 2014
 and was trained on the iBUG 300-W face landmark dataset.
 
 Also, note that you can train your own models using dlib's machine learning
 tools.  See train_shape_predictor_ex.cpp to see an example.
 
 
 
 
 Finally, note that the face detector is fastest when compiled with at least
 SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
 chip then you should enable at least SSE2 instructions.  If you are using
 cmake to compile this program you can enable them by using one of the
 following commands when you create the build project:
 cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
 cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
 cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
 This will set the appropriate compiler options for GCC, clang, Visual
 Studio, or the Intel compiler.  If you are using another compiler then you
 need to consult your compiler's manual to determine how to enable these
 instructions.  Note that AVX is the fastest but requires a CPU from at least
 2011.  SSE4 is the next fastest and is supported by most current machines.
 */


#include <dlib\image_processing\frontal_face_detector.h>
#include <dlib\image_processing\render_face_detections.h>
#include <dlib\image_processing.h>
#include <dlib\gui_widgets.h>
#include <dlib\image_io.h>
#include <dlib\opencv.h>
#include "Dlib_facedetection.h"
#include <iostream>

#include "opencv2\objdetect.hpp"
#include "opencv2\videoio.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include <stdio.h>


using namespace cv;
using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

int main()
{
    cv::VideoCapture cap("C:\\Users\\xli252\\Desktop\\VTS_01_2.VOB");
    cv::Mat Image;
    string dat="C:\\Users\\xli252\\Desktop\\Landmark\\test\\shape_predictor_68_face_landmarks.dat";
    std::vector<Point2f> b;
    int i = 0;
    //Dlib_facedetection a;//face detection
    Dlib_facedetection a(dat);//face landmark
    while (cap.read(Image)) {
        if(i%30 == 0)
        {
            //resize(Image, Image, Image.size()/2);            
            //uint64_t start_time = mach_absolute_time();
            a.loadImg(Image);
            //uint64_t end_time = mach_absolute_time();
            //uint64_t elapsed = 1e-6*(end_time - start_time);
            //resize(Image, Image, Image.size()*2);
            b = a.faceHOG;           
        }
        //cv::Mat Image2;
        
        //cvtColor(Image,Image,CV_RGB2GRAY);
        
        
        for(int i= 0; i < b.size(); i=+2)
        {
            cv::rectangle(Image, b[i], b[i+1], Scalar(255,0,255));
        }
        
        
        //face landmark
        
        if(a.dataVisiable == 1)
        {
            for(int i=0;i<a.faceLandmarks.size();i++)
            {
                for(int j=0; j<a.faceLandmarks[i].size();j++)
                {
                    circle(Image, a.faceLandmarks[i][j], 1,Scalar(255,255,0));
                }
            }
        }
        
        i++;
        
        imshow("a",Image);
        waitKey(1);
    }
    
    //Image = imread("/Users/lixile/Desktop/xile.jpg");
    
}

