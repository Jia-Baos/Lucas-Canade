#include "AffineEstimate.h"

/*******************Class of ImageProcessor*******************/

ImageProcessor::ImageProcessor()
{
	// define the mask to compute gradient
	kx_ = (cv::Mat_<double>(3, 3) << 0, 0, 0, -0.5, 0, 0.5, 0, 0, 0);
	ky_ = (cv::Mat_<double>(3, 3) << 0, -0.5, 0, 0, 0, 0, 0, 0.5, 0);
}

bool ImageProcessor::setInput(const cv::Mat& image)
{
	if (image.empty()) return false;

	// change the BGR to Gray
	cv::Mat gray;

	// asset image is BGR or Gray
	if (image.type() == CV_8UC3)
	{
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	}
	else
	{
		gray = image;
	}

	// change the type of data
	gray.convertTo(image_, CV_64FC1);

	return true;
}

void ImageProcessor::getGradient(cv::Mat& gx, cv::Mat& gy) const
{
	if (image_.empty())
	{

		std::cerr << "Input image is empty, please use setInput() before call function getGradient()" << std::endl;
		return;
	}

	// compute the gradient of image_
	cv::filter2D(image_, gx, -1, kx_);
	cv::filter2D(image_, gy, -1, ky_);

}

double ImageProcessor::getBilinearInterpolation(const cv::Mat& image, double x, double y)
{
	if (image.empty() || image.type() != CV_64FC1)
	{

		std::cerr << "Input image is empty, please use setInput() before call function getGradient()" << std::endl;
		return -1;
	}

	int row = (int)y;
	int col = (int)x;

	double rr = y - row;
	double cc = x - col;

	// Bilinear Interpolation and get the value of pixel
	return (1 - rr) * (1 - cc) * image.at<double>(row, col) +
		(1 - rr) * cc * image.at<double>(row, col + 1) +
		rr * (1 - cc) * image.at<double>(row + 1, col) +
		rr * cc * image.at<double>(row + 1, col + 1);
}

double ImageProcessor::getBilinearInterpolation(double x, double y) const
{
	return getBilinearInterpolation(image_, x, y);
}


/*******************Class of AffineEstimator*******************/

AffineEstimator::AffineEstimator() : max_iter_(80), debug_show_(true)
{
	image_processor_ = new ImageProcessor;
}

AffineEstimator::~AffineEstimator()
{
	if (image_processor_ != nullptr)
		delete image_processor_;
}

// compute the points of region which has been affined
std::array<cv::Point2d, 4> affinedRectangle(const AffineEstimator::AffineParameter& affine, const cv::Rect2d& rect)
{
	std::array<cv::Point2d, 4> result;

	result[0] = cv::Point2d((1 + affine.p1) * rect.x + affine.p3 * rect.y + affine.p5,
		affine.p2 * rect.x + (1 + affine.p4) * rect.y + affine.p6);

	result[1] = cv::Point2d((1 + affine.p1) * rect.x + affine.p3 * (rect.y + rect.height) + affine.p5,
		affine.p2 * rect.x + (1 + affine.p4) * (rect.y + rect.height) + affine.p6);

	result[2] = cv::Point2d((1 + affine.p1) * (rect.x + rect.width) + affine.p3 * (rect.y + rect.height) + affine.p5,
		affine.p2 * (rect.x + rect.width) + (1 + affine.p4) * (rect.y + rect.height) + affine.p6);

	result[3] = cv::Point2d((1 + affine.p1) * (rect.x + rect.width) + affine.p3 * rect.y + affine.p5,
		affine.p2 * (rect.x + rect.width) + (1 + affine.p4) * rect.y + affine.p6);

	return result;
}

void AffineEstimator::debugShow()
{
	auto points = affinedRectangle(affine_, cv::Rect2d(0, 0, tx_.cols, tx_.rows));

	cv::Mat imshow = imshow_.clone();

	// the region has been affined
	cv::line(imshow, points[0], points[1], cv::Scalar(0, 0, 255));
	cv::line(imshow, points[1], points[2], cv::Scalar(0, 0, 255));
	cv::line(imshow, points[2], points[3], cv::Scalar(0, 0, 255));
	cv::line(imshow, points[3], points[0], cv::Scalar(0, 0, 255));

	// the orginal region
	cv::rectangle(imshow, cv::Rect(32, 52, 100, 100), cv::Scalar(0, 255, 0));

	cv::namedWindow("debug show", cv::WINDOW_NORMAL);
	cv::imshow("debug show", imshow);
}

// the entry of main function
void AffineEstimator::compute(const cv::Mat& source_image, const cv::Mat& template_image, const AffineParameter& affine_init, const Method& method)
{

	// copy(deep) the affine matrix to another memory
	memcpy(affine_.data, affine_init.data, sizeof(double) * 6);

	// change the bgr to gray
	image_processor_->setInput(source_image);
	template_image.convertTo(tx_, CV_64FC1);

	if (debug_show_)
		cv::cvtColor(source_image, imshow_, cv::COLOR_GRAY2BGR);

	switch (method)
	{
	case Method::kForwardAdditive:
		computeFA();
		break;

	case Method::kForwardCompositional:
		//computeFC();
		break;

	case Method::kBackwardAdditive:
		//computeBA();
		break;

	case Method::kBackwardCompositional:
		//computeBC();
		break;

	default:
		std::cerr << "Invalid method type, please check." << std::endl;
		break;
	}
}

void AffineEstimator::computeFA()
{
	// affine matrix
	cv::Mat p = cv::Mat(6, 1, CV_64FC1, affine_.data);

	// compute the gradient
	cv::Mat gx, gy;
	image_processor_->getGradient(gx, gy);

	int i = 0;
	for (; i < max_iter_; ++i)
	{

		if (debug_show_)
			debugShow();

		cv::Mat hessian = cv::Mat::zeros(6, 6, CV_64FC1);
		cv::Mat residual = cv::Mat::zeros(6, 1, CV_64FC1);

		double cost = 0.;

		for (int y = 0; y < tx_.rows; y++)
		{
			for (int x = 0; x < tx_.cols; x++)
			{
				// new coordinate of every pixel of template
				double wx = (double)x * (1. + affine_.p1) + (double)y * affine_.p3 + affine_.p5;
				double wy = (double)x * affine_.p2 + (double)y * (1. + affine_.p4) + affine_.p6;

				if (wx < 1 || wx > image_processor_->width() - 2 || wy < 1 || wy > image_processor_->height() - 2)
					continue;

				double i_warped = image_processor_->getBilinearInterpolation(wx, wy);

				// the arr of value of pixel
				double err = i_warped - tx_.at<double>(y, x);

				double gx_warped = image_processor_->getBilinearInterpolation(gx, wx, wy);
				double gy_warped = image_processor_->getBilinearInterpolation(gy, wx, wy);

				cv::Mat jacobian = (cv::Mat_<double>(1, 6) << x * gx_warped, x * gy_warped,
					y * gx_warped, y * gy_warped, gx_warped, gy_warped);

				cv::Mat jacobianTmp;
				cv::transpose(jacobian, jacobianTmp);
				hessian += jacobianTmp * jacobian;
				residual -= jacobianTmp * err;

				cost += err * err;
			}
		}

		cv::Mat hessianTmp;
		cv::invert(hessian, hessianTmp, cv::DECOMP_CHOLESKY);
		cv::Mat delta_p = hessianTmp * residual;
		p += delta_p;

		std::cout << "Iteration " << i << " cost = " << cost << " squared delta p L2 norm = " << cv::norm(delta_p, cv::NORM_L2) << std::endl;

		if (cv::norm(delta_p, cv::NORM_L2) < 1e-12)
			break;
	}

	std::cout << "After " << i + 1 << " iteration, the final estimate affine matrix is: \n"
		<< affine_.p1 + 1 << " " << affine_.p3 << " " << affine_.p5 << " \n"
		<< affine_.p2 << " " << affine_.p4 + 1 << " " << affine_.p6 << std::endl;
}
