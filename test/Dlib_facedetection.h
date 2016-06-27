//
//  Dlib_facedetection.hpp
//  examples
//
//  Created by LiXile  on 2016-06-25.
//
//

#ifndef Dlib_facedetection_hpp
#define Dlib_facedetection_hpp

#include <dlib\image_processing\frontal_face_detector.h>
#include <dlib\image_processing\render_face_detections.h>
#include <dlib\image_processing.h>
#include <iostream>

#include <dlib\opencv.h>
#include "opencv2\objdetect.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\core.hpp"
#include <stdio.h>


//#include <mach\mach_time.h>
//#include <time.h>
using namespace cv;
using namespace dlib;
using namespace std;


class Dlib_facedetection
{
public:
    //constructor
    Dlib_facedetection(); //face detection
    Dlib_facedetection(string datName); //face landmark
    Dlib_facedetection(cv::Mat img, string datName);
    
    //for video
    void loadImg(cv::Mat img);
    
    cv_image<bgr_pixel> image_transfer(cv::Mat img);
    //face detection
    std::vector<Point2f> Dlib_ft();
    std::vector<Point2f> Dlib_ft(cv::Mat img);
    std::vector<Point2f> Dlib_ft(cv_image<bgr_pixel> cimg);
    //face landmark
    std::vector<std::vector<Point2f> > Dlib_landmark();
    std::vector<std::vector<Point2f> > Dlib_landmark(cv::Mat img, string datName);
    std::vector<std::vector<Point2f> > Dlib_landmark(cv_image<bgr_pixel> cimg, string datName);

    
    //variables
    cv_image<bgr_pixel> cimg;
    std::vector<dlib::rectangle> detections;
    std::vector<Point2f> faceHOG;
    std::vector<std::vector<Point2f> > faceLandmarks;
   // uint64_t elapsed;
    
    //init variables
    frontal_face_detector detector;
    shape_predictor pose_model;
    bool dataVisiable = 0;
	int height = 0;
	int width = 0;
};

#endif /* Dlib_facedetection_hpp */

