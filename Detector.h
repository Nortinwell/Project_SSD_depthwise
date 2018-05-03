#include <caffe/caffe.hpp>
//#ifdef USE_OPENCV
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
////#include <opencv2/contrib/contrib.hpp>
//#endif  // USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost\smart_ptr.hpp>


using namespace caffe;
using namespace cv;
using namespace std;

class Detector {
public:
	Detector();
	void Init(const string& model_file,
		const string& weights_file,
		const string& mean_file,
		const string& mean_value);

	std::vector<vector<float> > Detect(const cv::Mat& img);

private:
	void SetMean(const string& mean_file, const string& mean_value);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

private:
	boost::shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
};