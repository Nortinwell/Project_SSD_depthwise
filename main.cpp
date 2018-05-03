//#include "Classifier.h"
#include "Detector.h"
#include "caffe_layers.h"
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
				rect_vec[j] = rect_vec[j+1];
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
void main()
{
	cout << "轮胎模型字符检测程序正式开始！" << endl;
	//system("pause");

	const float confidence_threshold = 0.05;
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

	string piclist = "./images/";
	char filename[32];
	int name ;
	cin >> name;
	sprintf_s(filename, "%d.png", name);

	double t = (double)getTickCount();   //计时

	Mat img = imread(piclist+filename);
	Mat dst = img.clone();
	std::vector<vector<float> > detections = detector.Detect(img);

	//边框结果
	std::vector<cv::Rect> rect_bc;
	std::vector<cv::Rect> rect_sc;

	//输出结果
	std::vector<std::pair<std::string, cv::Rect>> plateResult;

	//没检测到车牌
	if (!detections.size())
		return;

	char score_s[10];
	for (int i = 0; i < detections.size(); ++i) 
	{
		const vector<float>& d = detections[i];
		// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
		CHECK_EQ(d.size(), 7);

		//检测到的车牌置信率
		const int label = d[1];
		cout << "类别：" << label << endl;
		const float score = d[2];
		if (score >= confidence_threshold)
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
			if (label == 1)
			{
				rect_bc.push_back(cv::Rect(xmin, ymin, width , height ));
				rectangle(dst, Rect(xmin, ymin, width , height ), Scalar(0, 155, 255), 2, 8);
			//	cout << "类别：" << label << "   置信度：" << score << endl;
				sprintf_s(score_s, "%.2f", score);
				putText(dst, score_s, Point(xmin, ymin), FONT_HERSHEY_PLAIN,2,Scalar(125,125,255),2,8);
			}
			else
			{
				rect_sc.push_back(cv::Rect(xmin, ymin, width , height ));
				rectangle(dst, Rect(xmin, ymin, width , height ), Scalar(125, 0, 255), 2, 8);
			//	cout << "类别：" << label << "   置信度：" << score << endl;
				sprintf_s(score_s, "%.2f", score);
				putText(dst, score_s, Point(xmin, ymin), FONT_HERSHEY_PLAIN, 2, Scalar(125, 125, 255), 2, 8);
			}
		}
	}

	t = double((getTickCount() - t) / getTickFrequency());
	cout << "检测一共花费时间为" << t << endl;

	imshow("检测图", dst);
	waitKey(1000);

	cout << "rect_sc'size: " << rect_sc.size() << endl;
	cout << endl;
	for (int i = 0; i < rect_sc.size(); i++)
		cout << "rect_sc'y : " << rect_sc[i].y << endl;
	cout << endl;

//	vector<Rect> rect1, rect2, rect3, rect4;
	vector<vector<Rect>> rect_vec;
	rect_vec = sc_sort(rect_sc);

	for (int i = 0; i < rect_vec.size(); i++)
	{
		for (int j = 0; j < rect_vec[i].size(); j++)
			cout << "rect1 : " << rect_vec[i][j] << endl;
		cout << endl;
	}


//	cout << "rect1.size():  " << rect1.size() << "   rec2.size():  " << rect2.size() << "   rect3.size():  " << rect3.size() << "   rect4.size():  " << rect4.size() << endl;

	
	system("pause");

}