// C_camshift.cpp : 定義主控台應用程式的進入點。
//

#include "stdafx.h"
#include "cv.h" 
#include "highgui.h"

//#include "opencv2/video/tracking.hpp"
//#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
class CamShiftPatch
{
public:
	Rect TrackWindow;

	int vmin = 10;
	int vmax = 256;
	int smin = 30;



	CamShiftPatch()
	{
		TrackWindow = cvRect(0, 0, 50, 50);
	}

	~CamShiftPatch()
	{
	}

	Mat CreatHist(Mat image, Rect ROIwindow, OutputArray histData)
	{
		Mat Image_hsv;
		cvtColor(image, Image_hsv, COLOR_BGR2HSV);

		Mat mask;
		//目的為把 S值小(接近白色) 和 V值小(接近黑色) 的兩個地方濾掉
		inRange(
			Image_hsv,
			Scalar(0, smin, MIN(vmin, vmax)),
			Scalar(180, 256, MAX(vmin, vmax)),
			mask);

		Mat hueForHis;
		int ch[] = { 0, 0 };
		hueForHis.create(Image_hsv.size(), Image_hsv.depth());
		mixChannels(&Image_hsv, 1, &hueForHis, 1, ch, 1);//把Image_hsv中的Hue值拿出來給"hue"

		//建立 roi
		Mat roi(hueForHis, ROIwindow);
		Mat maskroi(mask, ROIwindow);
		int hsize = 16;

		Mat histData_out;
		float hranges[] = { 0,180 };
		const float* phranges = hranges;
		calcHist(&roi, 1, 0, maskroi, histData_out, 1, &hsize, &phranges);
		normalize(histData_out, histData_out, 0, 255, CV_MINMAX);

		//畫直方圖
		Mat histImg = Mat::zeros(200, 320, CV_8UC3);
		histImg = Scalar::all(0);
		int binW = histImg.cols / hsize;
		Mat buf(1, hsize, CV_8UC3);
		for (int i = 0; i < hsize; i++)
			buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180. / hsize), 255, 255);
		cvtColor(buf, buf, CV_HSV2BGR);

		for (int i = 0; i < hsize; i++)
		{
			int val = saturate_cast<int>(histData_out.at<float>(i)*histImg.rows / 255);
			rectangle(histImg, Point(i*binW, histImg.rows),
				Point((i + 1)*binW, histImg.rows - val),
				Scalar(buf.at<Vec3b>(i)), -1, 8);
		}


		histData_out.copyTo(histData);
		return histImg;
	}

	RotatedRect GetTrackBox(Mat image,InputArray histData)
	{
		//1. 用image提取出Hue的圖像
		Mat Image_hsv;
		cvtColor(image, Image_hsv, COLOR_BGR2HSV);

		Mat hueForTrack;
		int ch[] = { 0, 0 };
		hueForTrack.create(Image_hsv.size(), Image_hsv.depth());
		mixChannels(&Image_hsv, 1, &hueForTrack, 1, ch, 1);//把Image_hsv中的Hue值拿出來給"hue"

		//2. 用Hue圖像 跟 histData做出 backproject
		Mat backproject;
		float hranges[] = { 0,180 };
		const float* phranges = hranges;
		calcBackProject(&hueForTrack, 1, 0, histData, backproject, &phranges);

		//3. backproject 要除去一些原圖接近黑色或白色的地方
		Mat mask;
		inRange(	//目的為把 S值小(接近白色) 和 V值小(接近黑色) 的兩個地方濾掉
			Image_hsv,
			Scalar(0, smin, MIN(vmin, vmax)),
			Scalar(180, 256, MAX(vmin, vmax)),
			mask);
		backproject &= mask;//相加給backproject ->就是用把backproject放上mask

		//4. 做CamShift(會需要packproject，會得到trackWindow)
		RotatedRect trackBox = CamShift(backproject, TrackWindow,
			TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));

		//如果找到太小或沒有的話
		if (TrackWindow.area() <= 1)
		{
			int cols = backproject.cols, rows = backproject.rows, r = (MIN(cols, rows) + 5) / 6;
			TrackWindow = Rect(TrackWindow.x - r, TrackWindow.y - r,
				TrackWindow.x + r, TrackWindow.y + r) &
				Rect(0, 0, cols, rows);
		}

		return trackBox;
	}
};


/*  
*    Main
*/

Rect selection;
static void onMouse(int event, int x, int y, int, void*)
{
	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
		selection.x = x;
		selection.y = y;
		break;
	case CV_EVENT_LBUTTONUP:
		selection.width = x- selection.x;
		selection.height = y- selection.y;

		break;
	}
}

int main()
{
	const string name_CamShift = "CAMshift";
	const string name_hist = "Histogram";
	VideoCapture capture(0);

	CamShiftPatch myCAMShift;
	namedWindow(name_CamShift);
	namedWindow(name_hist);
	setMouseCallback(name_CamShift, onMouse, 0);
	Mat frame;  //定義一個Mat變數，用于存儲每一幅的圖像
	Mat hisImg;

	Mat HistDataSelect;
	while (1)
	{
		//先找ROI
		capture >> frame;
		
		//★★先做出histogram★★
		hisImg = myCAMShift.CreatHist(frame, selection, HistDataSelect);
		//★★可以先設定初始的TrackWindow會追得比較準★★
		myCAMShift.TrackWindow = selection;
		

		rectangle(frame, selection, cvScalar(255, 50, 50), 3);
		imshow(name_CamShift, frame);
	
		if ((char)waitKey(10) == 's')
			break;
	}

	imshow(name_hist, hisImg);

	while (1)
	{
		capture >> frame;
		//★★做CamShift得到trackBox★★
		RotatedRect trackBox =myCAMShift.GetTrackBox(frame,HistDataSelect);
		
		Mat out;
		frame.copyTo(out);
		ellipse(out, trackBox, Scalar(0, 0, 255), 3, CV_AA);
		imshow(name_CamShift, out);

		char c = (char)waitKey(10);
		if (c == 27) //esc
			break;
	}
    return 0;
}





