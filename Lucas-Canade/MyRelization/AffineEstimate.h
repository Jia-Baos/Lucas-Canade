#pragma once
#ifndef AFFINEESTIMATE_H
#define AFFINEESTIMATE_H

#include <iostream>
#include <array>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "Common.h"

class ImageProcessor {
public:
	ImageProcessor();
	~ImageProcessor() = default;

	bool setInput(const cv::Mat& image);

	// the const means this function will not change the Member variables
	int width() const { return image_.cols; }
	int height() const { return image_.rows; }

	void getGradient(cv::Mat& gx, cv::Mat& gy) const;

	double getBilinearInterpolation(double x, double y) const;

	// the static means all objects of the class share this single member function
	static double getBilinearInterpolation(const cv::Mat& image, double x, double y);



private:
	// image_ is the copy of image with type is CV_64FC1
	cv::Mat image_;
	cv::Mat kx_, ky_;
};


class AffineEstimator {

public:
	union AffineParameter
	{
		double data[6];
		struct {
			double p1, p2, p3, p4, p5, p6;
		};
	};

	AffineEstimator();
	~AffineEstimator();

	// align a template image to source image using an affine transformation.
	void compute(const cv::Mat& source_image, const cv::Mat& template_image,
		const AffineParameter& affine_init, const Method& method = Method::kForwardAdditive);

private:

	void computeFA();
	/*void computeFC();
	void computeBA();
	void computeBC();*/

	// only for debug show
	void debugShow();

	ImageProcessor* image_processor_;

	bool debug_show_;

	// tx_ is the copy of tempate_image with type is CV_64FC1
	cv::Mat tx_;
	cv::Mat imshow_;

	// store the affine matrix
	AffineParameter affine_;

	// maximum iteration number
	int max_iter_;

};
#endif // !AFFINEESTIMATE_H