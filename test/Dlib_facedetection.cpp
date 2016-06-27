//
//  Dlib_facedetection.cpp
//  examples
//
//  Created by LiXile  on 2016-06-25.
//
//

#include "Dlib_facedetection.h"

Dlib_facedetection::Dlib_facedetection()
{
    detector = get_frontal_face_detector();
}

Dlib_facedetection::Dlib_facedetection(string datName)
{
    detector = get_frontal_face_detector();
    deserialize(datName) >> pose_model;
    dataVisiable = 1;
}

Dlib_facedetection::Dlib_facedetection(cv::Mat img, string datName)
{
    cimg = image_transfer(img);
   // uint64_t start = mach_absolute_time();
    faceHOG = Dlib_ft(cimg);
   // uint64_t end_time = mach_absolute_time();
   // elapsed= 1e-9*(end_time-start);
//    faceLandmarks=Dlib_landmark(cimg, datName);
}

void Dlib_facedetection::loadImg(cv::Mat img)
{
    cimg = image_transfer(img);
    faceHOG = Dlib_ft();
    if(dataVisiable == 1)
    {
        faceLandmarks=Dlib_landmark();
    }
}

cv_image<bgr_pixel> Dlib_facedetection::image_transfer(cv::Mat img)
{
	height = img.size().height;
	width = img.size().width;
	cv_image<bgr_pixel> cimg(img);
    return cimg;
}

std::vector<Point2f> Dlib_facedetection::Dlib_ft(cv::Mat img)
{
    frontal_face_detector detector = get_frontal_face_detector();
    cv_image<bgr_pixel> cimg(img);
    //pyramid_up(cimg);
    std::vector<dlib::rectangle> dets = detector(cimg);
    std::vector<Point2f> Dlib_ft_result;
    for (int i = 0; i < dets.size(); i++)
    {
       //Dlib_ft_result[i]= (dets[0],dets[1]);
       Point2f leftTop(dets[i].l, dets[i].t);
       Point2f rightBottom(dets[i].r, dets[i].b);
       Dlib_ft_result.push_back(leftTop);
       Dlib_ft_result.push_back(rightBottom);
    }
    return Dlib_ft_result;
}

std::vector<Point2f> Dlib_facedetection::Dlib_ft(cv_image<bgr_pixel> cimg)
{
    frontal_face_detector detector = get_frontal_face_detector();
    detections = detector(cimg);
    std::vector<Point2f> Dlib_ft_result;
    for (int i = 0; i < detections.size(); i++)
    {
        //Dlib_ft_result[i]= (dets[0],dets[1]);
        Point2f leftTop(detections[i].l, detections[i].t);
        Point2f rightBottom(detections[i].r, detections[i].b);
        Dlib_ft_result.push_back(leftTop);
        Dlib_ft_result.push_back(rightBottom);
    }
    return Dlib_ft_result;
}

std::vector<Point2f> Dlib_facedetection::Dlib_ft()
{
    detections = detector(cimg);
    std::vector<Point2f> Dlib_ft_result;
    for (int i = 0; i < detections.size(); i++)
    {
        //Dlib_ft_result[i]= (dets[0],dets[1]);
		if (detections[i].l < 0)
			detections[i].l = 0;
		if (detections[i].t < 0)
			detections[i].t = 0;
		if (detections[i].r > width)
			detections[i].r = width;
		if (detections[i].b > height)
			detections[i].b = height;
        Point2f leftTop(detections[i].l, detections[i].t);
        Point2f rightBottom(detections[i].r, detections[i].b);
        Dlib_ft_result.push_back(leftTop);
        Dlib_ft_result.push_back(rightBottom);
    }
    return Dlib_ft_result;
}


std::vector<std::vector<Point2f> > Dlib_facedetection::Dlib_landmark(cv::Mat img, string datName)
{
    frontal_face_detector detector = get_frontal_face_detector();
    cv_image<bgr_pixel> cimg(img);
    std::vector<dlib::rectangle> dets = detector(cimg);
    
    shape_predictor pose_model;
    deserialize(datName) >> pose_model;

    std::vector<std::vector<Point2f> > landmarks;
    for (int j = 0; j < dets.size(); j++)
    {
        full_object_detection shape = pose_model(cimg, dets[j]);
        std::vector<Point2f> landmark;
        for (int i = 0; i < 68; i++)
        {
            Point2f a(shape.parts[i].x(), shape.parts[i].y());
            landmark.push_back(a);
        }
        landmarks.push_back(landmark);
    }
    return landmarks;

}


std::vector<std::vector<Point2f> > Dlib_facedetection::Dlib_landmark(cv_image<bgr_pixel> cimg, string datName)
{
    shape_predictor pose_model;
    deserialize(datName) >> pose_model;
    std::vector<std::vector<Point2f> > landmarks;
    for (int j = 0; j < detections.size(); j++)
    {
        full_object_detection shape = pose_model(cimg, detections[j]);
        std::vector<Point2f> landmark;
        for (int i = 0; i < 68; i++)
        {
            Point2f a(shape.parts[i].x(), shape.parts[i].y());
            landmark.push_back(a);
        }
        landmarks.push_back(landmark);
    }
    return landmarks;
    
}

std::vector<std::vector<Point2f> > Dlib_facedetection::Dlib_landmark()
{
    std::vector<std::vector<Point2f> > landmarks;
    for (int j = 0; j < detections.size(); j++)
    {
        full_object_detection shape = pose_model(cimg, detections[j]);
        std::vector<Point2f> landmark;
        for (int i = 0; i < 68; i++)
        {
            Point2f a(shape.parts[i].x(), shape.parts[i].y());
            landmark.push_back(a);
        }
        landmarks.push_back(landmark);
    }
    return landmarks;
    
}
//
//*/