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

	Mat image = imread("C:\\Users\\gu573\\Documents\\GitHub\\Computer Vision\\img\\Jani2.jpg", 0);//读取图像，注意\\表示\的转义
	if (image.empty())
	{

		printf("Could not load image...\n");
		return -1;
	}
	Mat imageS;//缩小后的图像
	Mat imageConverted;
	Mat imageS_Blur;//通过中值滤波后的图像
	Mat imageSobelx, imageSobely;//x,y方向使用Sobel算子卷积所得的图像
	Mat imageSobelxAbs, imageSobelyAbs;//x,y方向使用Sobel算子卷积所得的图像并取绝对值
	Mat imageSobel;//最终通过Sobel算子检测的边缘图像
	Mat Gx = (Mat_<int>(3, 3) <<// Sobel算子横（x）向处理的卷积核
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1);
	Mat Gy = (Mat_<int>(3, 3) <<// Sobel算子纵（y）向处理的卷积核
		-1, -2, -1,
		0, 0, 0,
		1, 2, 1);
	//cout << Gx << endl
	//	<< Gy << endl;

	resize(image, imageS, imageS.size(), 0.16, 0.2, 1);//改变图像大小，以完整显示
	medianBlur(imageS, imageS_Blur, 5);//对图像进行中值滤波，否则噪声太大效果不好(也可以用高斯滤波等等尝试一下哪个效果更好，在这里暂时使用中值滤波)。经实验卷积核越大，效果越好，取值范围1-5。
	filter2D(imageS_Blur,imageSobelx,CV_64F,Gx);//x方向卷积
	convertScaleAbs(imageSobelx, imageSobelxAbs);//需要取绝对值，否则卷积所得负值在显示的时候默认为0，将损失边界
	filter2D(imageS_Blur, imageSobely, CV_64F, Gy);//y方向卷积
	convertScaleAbs(imageSobely, imageSobelyAbs);//同理
	addWeighted(imageSobelxAbs, 0.5, imageSobelyAbs, 0.5, 0, imageSobel);//按权重将两个方向卷积所得图像融合
	//cout << imageS.size();
	imgShow("Jani", imageSobel);//输出最终结果
	return 0;

}
//显示图像
void imgShow(const String& imgName, InputArray img)
{
	imshow(imgName, img);
	waitKey(0);
	destroyAllWindows();
}
