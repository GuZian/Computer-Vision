/*
	edge detecion.cpp

	This demo is used to detect the edge of image.
	The algorithm is based on Sobel operator.
	I just used the package of Opencv to decode the image(read the image and convert it to matrix) and do some morphological processing.The principle of Sobel is coded by myself.

	by Zian Gu
	04/17/2020
*/


#include <opencv2/opencv.hpp>
#include<iostream>
using namespace cv;
using namespace std;
void imgShow(const String& imgName, InputArray img);

int main(int argc, char** argv) {

	Mat image = imread("C:\\Users\\gu573\\Documents\\GitHub\\Computer Vision\\img\\Jani2.jpg", 0);//��ȡͼ��ע��\\��ʾ\��ת��
	if (image.empty())
	{

		printf("Could not load image...\n");
		return -1;
	}
	Mat imageS;//��С���ͼ��
	Mat imageConverted;
	Mat imageS_Blur;//ͨ����ֵ�˲����ͼ��
	Mat imageSobelx, imageSobely;//x,y����ʹ��Sobel���Ӿ�����õ�ͼ��
	Mat imageSobelxAbs, imageSobelyAbs;//x,y����ʹ��Sobel���Ӿ�����õ�ͼ��ȡ����ֵ
	Mat imageSobel;//����ͨ��Sobel���Ӽ��ı�Եͼ��
	Mat Gx = (Mat_<int>(3, 3) <<// Sobel���Ӻᣨx������ľ����
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1);
	Mat Gy = (Mat_<int>(3, 3) <<// Sobel�����ݣ�y������ľ����
		-1, -2, -1,
		0, 0, 0,
		1, 2, 1);
	//cout << Gx << endl
	//	<< Gy << endl;

	resize(image, imageS, imageS.size(), 0.16, 0.2, 1);//�ı�ͼ���С����������ʾ
	medianBlur(imageS, imageS_Blur, 5);//��ͼ�������ֵ�˲�����������̫��Ч������(Ҳ�����ø�˹�˲��ȵȳ���һ���ĸ�Ч�����ã���������ʱʹ����ֵ�˲�)����ʵ������Խ��Ч��Խ�ã�ȡֵ��Χ1-5��
	filter2D(imageS_Blur,imageSobelx,CV_64F,Gx);//x������
	convertScaleAbs(imageSobelx, imageSobelxAbs);//��Ҫȡ����ֵ�����������ø�ֵ����ʾ��ʱ��Ĭ��Ϊ0������ʧ�߽�
	filter2D(imageS_Blur, imageSobely, CV_64F, Gy);//y������
	convertScaleAbs(imageSobely, imageSobelyAbs);//ͬ��
	addWeighted(imageSobelxAbs, 0.5, imageSobelyAbs, 0.5, 0, imageSobel);//��Ȩ�ؽ���������������ͼ���ں�
	//cout << imageS.size();
	imgShow("Jani", imageSobel);//������ս��
	return 0;

}
//��ʾͼ��
void imgShow(const String& imgName, InputArray img)
{
	imshow(imgName, img);
	waitKey(0);
	destroyAllWindows();
}
