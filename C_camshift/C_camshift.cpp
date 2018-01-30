// C_camshift.cpp : �w�q�D���x���ε{�����i�J�I�C
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
		//�ت����� S�Ȥp(����զ�) �M V�Ȥp(����¦�) ����Ӧa���o��
		inRange(
			Image_hsv,
			Scalar(0, smin, MIN(vmin, vmax)),
			Scalar(180, 256, MAX(vmin, vmax)),
			mask);

		Mat hueForHis;
		int ch[] = { 0, 0 };
		hueForHis.create(Image_hsv.size(), Image_hsv.depth());
		mixChannels(&Image_hsv, 1, &hueForHis, 1, ch, 1);//��Image_hsv����Hue�Ȯ��X�ӵ�"hue"

		//�إ� roi
		Mat roi(hueForHis, ROIwindow);
		Mat maskroi(mask, ROIwindow);
		int hsize = 16;

		Mat histData_out;
		float hranges[] = { 0,180 };
		const float* phranges = hranges;
		calcHist(&roi, 1, 0, maskroi, histData_out, 1, &hsize, &phranges);
		normalize(histData_out, histData_out, 0, 255, CV_MINMAX);

		//�e�����
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
		//1. ��image�����XHue���Ϲ�
		Mat Image_hsv;
		cvtColor(image, Image_hsv, COLOR_BGR2HSV);

		Mat hueForTrack;
		int ch[] = { 0, 0 };
		hueForTrack.create(Image_hsv.size(), Image_hsv.depth());
		mixChannels(&Image_hsv, 1, &hueForTrack, 1, ch, 1);//��Image_hsv����Hue�Ȯ��X�ӵ�"hue"

		//2. ��Hue�Ϲ� �� histData���X backproject
		Mat backproject;
		float hranges[] = { 0,180 };
		const float* phranges = hranges;
		calcBackProject(&hueForTrack, 1, 0, histData, backproject, &phranges);

		//3. backproject �n���h�@�ǭ�ϱ���¦�Υզ⪺�a��
		Mat mask;
		inRange(	//�ت����� S�Ȥp(����զ�) �M V�Ȥp(����¦�) ����Ӧa���o��
			Image_hsv,
			Scalar(0, smin, MIN(vmin, vmax)),
			Scalar(180, 256, MAX(vmin, vmax)),
			mask);
		backproject &= mask;//�ۥ[��backproject ->�N�O�Χ�backproject��Wmask

		//4. ��CamShift(�|�ݭnpackproject�A�|�o��trackWindow)
		RotatedRect trackBox = CamShift(backproject, TrackWindow,
			TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));

		//�p�G���Ӥp�ΨS������
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
	Mat frame;  //�w�q�@��Mat�ܼơA�Τ_�s�x�C�@�T���Ϲ�
	Mat hisImg;

	Mat HistDataSelect;
	while (1)
	{
		//����ROI
		capture >> frame;
		
		//���������Xhistogram����
		hisImg = myCAMShift.CreatHist(frame, selection, HistDataSelect);
		//�����i�H���]�w��l��TrackWindow�|�l�o����ǡ���
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
		//������CamShift�o��trackBox����
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





