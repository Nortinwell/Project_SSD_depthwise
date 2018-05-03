//#include "Classifier.h"
#include "Detector.h"
/*
layers register
*/
#include "caffe/common.hpp"
#include "caffe/layers/input_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/dropout_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/reshape_layer.hpp"
#include "caffe/layers/normalize_layer.hpp"
#include "caffe/layers/permute_layer.hpp"
#include "caffe/layers/flatten_layer.hpp"
#include "caffe/layers/prior_box_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/detection_output_layer.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/layers/deconv_layer.hpp"
namespace caffe
{
	extern INSTANTIATE_CLASS(InputLayer);
	extern INSTANTIATE_CLASS(InnerProductLayer);
	extern INSTANTIATE_CLASS(DropoutLayer);
	extern INSTANTIATE_CLASS(ConvolutionLayer);
	REGISTER_LAYER_CLASS(Convolution);
	extern INSTANTIATE_CLASS(ReLULayer);
	REGISTER_LAYER_CLASS(ReLU);
	extern INSTANTIATE_CLASS(PoolingLayer);
	REGISTER_LAYER_CLASS(Pooling);
	extern INSTANTIATE_CLASS(LRNLayer);
	REGISTER_LAYER_CLASS(LRN);
	extern INSTANTIATE_CLASS(SoftmaxLayer);
	REGISTER_LAYER_CLASS(Softmax);

	extern INSTANTIATE_CLASS(NormalizeLayer);
	//REGISTER_LAYER_CLASS(Normalize);
	extern INSTANTIATE_CLASS(PermuteLayer);
	//REGISTER_LAYER_CLASS(Permute);
	extern INSTANTIATE_CLASS(FlattenLayer);
	//REGISTER_LAYER_CLASS(Flatten);
	extern INSTANTIATE_CLASS(PriorBoxLayer);
	//REGISTER_LAYER_CLASS(PriorBox);
	extern INSTANTIATE_CLASS(ReshapeLayer);
	//REGISTER_LAYER_CLASS(Reshape);
	extern INSTANTIATE_CLASS(ConcatLayer);
	//REGISTER_LAYER_CLASS(Concat);
	extern INSTANTIATE_CLASS(DetectionOutputLayer);
	//REGISTER_LAYER_CLASS(DetectionOutput);
	extern INSTANTIATE_CLASS(BatchNormLayer);
	extern INSTANTIATE_CLASS(DeconvolutionLayer);
	//REGISTER_LAYER_CLASS(Deconvolution);

}
using namespace std;

#define h 20

vector<vector<Rect>> y_sort(vector<vector<Rect>> rect_vec)
{
	vector<Rect> tempRect;
	for (int i = 0; i < rect_vec.size(); i++)
	{
		for (int j = 0; j < rect_vec.size() - 1 - i; j++)
			if (rect_vec[j][0].y > rect_vec[j + 1][0].y)
			{
				tempRect = rect_vec[j];
				rect_vec[j] = rect_vec[j + 1];
				rect_vec[j + 1] = tempRect;
			}
	}
	return rect_vec;
}

vector<Rect> x_sort(vector<Rect> rect_x)
{
	Rect temp;
	//	int tempx, tempy, tempw, temph;
	for (int i = 0; i < rect_x.size(); i++)
	{
		for (int j = 0; j < rect_x.size() - 1 - i; j++)
		{
			if (rect_x[j].x > rect_x[j + 1].x)
			{
				temp = rect_x[j];
				rect_x[j] = rect_x[j + 1];
				rect_x[j + 1] = temp;
			}
		}
	}
	return rect_x;
}

vector<vector<Rect>> sc_sort(vector<Rect> rect_sc)
{
	vector<vector<Rect>> rect_vec;
	if (rect_sc.size() > 0)
	{
		vector<Rect> a1, a2, a3, a4;
		for (int i = 0; i < rect_sc.size(); i++)
		{
			if (abs(rect_sc[i].y - rect_sc[0].y) < h)
				a1.push_back(rect_sc[i]);
			else
				a2.push_back(rect_sc[i]);
		}

		if (a2.size() > 0)
		{
			vector<Rect> a2_copy = a2;
			a2.clear();
			for (int j = 0; j < a2_copy.size(); j++)
			{
				if (abs(a2_copy[j].y - a2_copy[0].y) < h)
					a2.push_back(a2_copy[j]);
				else
					a3.push_back(a2_copy[j]);
			}

			if (a3.size() > 0)
			{
				vector<Rect> a3_copy = a3;
				a3.clear();
				for (int k = 0; k < a3_copy.size(); k++)
				{
					if (abs(a3_copy[k].y - a3_copy[0].y) < h)
						a3.push_back(a3_copy[k]);
					else
						a4.push_back(a3_copy[k]);
				}

			}
		}
		cout << "rect1.size():  " << a1.size() << "   rec2.size():  " << a2.size() << "   rect3.size():  " << a3.size() << "   rect4.size():  " << a4.size() << endl;
		if (a1.size()>0) rect_vec.push_back(a1);
		if (a2.size()>0) rect_vec.push_back(a2);
		if (a3.size()>0) rect_vec.push_back(a3);
		if (a4.size()>0) rect_vec.push_back(a4);
	}
	//	cout << "rect_vec.size(): " << rect_vec.size() << endl;
	rect_vec = y_sort(rect_vec);
	//	cout << "rect.size():  " << rect_vec.size() << endl;;
	for (int i = 0; i < rect_vec.size(); i++)
	{
		rect_vec[i] = x_sort(rect_vec[i]);
		//cout << "rect1 :  " << rect_vec[i][0] << "   rect2 :  " << rect_vec[i][1] << endl;
	}

	return rect_vec;
}



Detector detector;
// Classifier rec_classifier;


void ssd_detect(Mat img, int str)
{
//	cvtColor(img, img, COLOR_BGR2RGB);
	Mat dst = img.clone();
	std::vector<vector<float> > detections = detector.Detect(img);

	char filename[30];
	sprintf_s(filename, "%d.txt", str);
	ofstream file(filename, ios::out);//打开一个文件,等同于file.open("test.txt", ios::out);

	for (int i = 0; i < detections.size(); ++i)
	{
		const vector<float>& d = detections[i];
		// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
		CHECK_EQ(d.size(), 7);

		//检测到的字符置信率
		const int label = d[1];
		const float score = d[2];

		if (label == 1 && score >= 0.1)
		{
			int width, height;
			//放大边框
			int xmin, ymin, xmax, ymax;

			xmin = int(d[3] * img.cols);
			if (xmin <= 15) xmin = 0;
			ymin = int(d[4] * img.rows);
			if (ymin < 0) ymin = 0;
			xmax = int(d[5] * img.cols);
			if (xmax > img.cols) xmax = img.cols;
			ymax = int(d[6] * img.rows);
			if (ymax > img.rows) ymax = img.rows;

			width = xmax - xmin;
			height = ymax - ymin;
			Rect object_bc(xmin, ymin, width, height);

			rectangle(dst, object_bc, Scalar(0, 155, 255), 2, 8);
			file << xmin << "," << ymin << "," << xmax << "," << ymax << "," <<"bc"<< endl;

		}

		if (label == 2 && score >= 0.1)
		{
			int width, height;
			//放大边框
			int xmin, ymin, xmax, ymax;

			xmin = int(d[3] * img.cols);
			if (xmin < 0) xmin = 0;
			ymin = int(d[4] * img.rows);
			if (ymin < 0) ymin = 0;
			xmax = int(d[5] * img.cols);
			if (xmax > img.cols) xmax = img.cols;
			ymax = int(d[6] * img.rows);
			if (ymax > img.rows) ymax = img.rows;

			width = xmax - xmin;
			height = ymax - ymin;
			Rect object_sc(xmin, ymin, (width), height);

			rectangle(dst, object_sc, Scalar(0, 155, 255), 2, 8);
			file << xmin << "," << ymin << "," << xmax << "," << ymax << "," << "sc" << endl;

		}
	}
	file.close();
	//imshow("dst", dst);
	//waitKey();
}


vector<string> ReadListImage(ifstream &img_data)
{
	vector<string> img_path;
	img_path.clear();
	string buf;
	char line[512];
	while (img_data)
	{
		if (getline(img_data, buf))
			img_path.push_back(buf);
	}
	img_data.close();
	return img_path;
}

void main()
{
	cout << "轮胎模型字符检测程序正式开始！" << endl;
	//system("pause");
	cout << "模型初始化..." << endl;
	Caffe::set_mode(Caffe::GPU);
	string folderName = "models";

	//检测模型初始化
	string model = folderName + "/detection.prototxt";
	string weights = folderName + "/detection.caffemodel";
	string mean = "104,117,123";
	const string& model_file = model;
	const string& weights_file = weights;
	const string& mean_file = "";
	const string& mean_value = mean;

	// Initialize the detection network.
	detector.Init(model_file, weights_file, mean_file, mean_value);

	// Initialize the recognition network.
	//	rec_classifier.Init(rec_model, rec_weights, labels);
	cout << "模型初始化完毕！" << endl;

	vector<string> RealPicPath;
	ifstream RealPicData("pic_list.txt");
	RealPicPath = ReadListImage(RealPicData);

	cout << "size: " << RealPicPath.size() << endl;
//	cout << "path: " << RealPicPath[0] << endl;
//	double t = (double)getTickCount();   //计时
	for (int i = 0; i<RealPicPath.size(); i++)   //因为图像是反向拍摄，所以要倒读才能更好的使用前项信息
	{
		Mat src =imread(RealPicPath[i], 1);
	/*	imshow("src",src);
		waitKey();*/
		cout << "处理的图像名字： " << i+360 << ".png " << endl;
		ssd_detect(src, i+360);

	}
//	t = double((getTickCount() - t) / getTickFrequency());
//	cout << "检测一共花费时间为" << t << endl;


	system("pause");

}