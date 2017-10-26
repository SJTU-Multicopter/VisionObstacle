#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <math.h>
#include "/home/clarence/catkin_ws/devel/include/Eigen/Dense"  

using namespace std;
using namespace cv;
using namespace Eigen;

#define WIDTH 640
#define HEIGHT 480
#define STEP 5
#define VALID_THRE 10

Mat dep_frame(WIDTH,HEIGHT,CV_16UC1);

int square_nw, square_nh, square_num;
//sensor_msgs::ImagePtr msg;

//camera parameters
float fx = 525.0;
float fy = 525.0;
float cx = 319.5;
float cy = 239.5;
float factor = 5000.0;

Vector4f &plane_ransac(MatrixXf &square, Vector4f &paras, Vector3f &plane_vec, Vector3f &plane_point, int &pixel_counter, int iterations = 10, float dist_filter = 0.2f, float ratio = 1.f) //Use RANSAC to fit planes. MatrixXf square(STEP*STEP, 3), (X,Y,Z) for each pixel
{
	int n1, n2, n3;
	VectorXf dist(pixel_counter);
	Vector4f paras_temp;
	float error_min = 10000.f;

	for(int i = 0; i < iterations; i++)
	{
		Vector3f n = Vector3f::Random();  //will the values be the same???
		n1 = (int)((n(0) + 1) / 2 * pixel_counter);
		n2 = (int)((n(1) + 1) / 2 * pixel_counter);
		n3 = (int)((n(2) + 1) / 2 * pixel_counter);

		//a, b, c, d paras calculate
		paras_temp(0) = (square(n2,1)-square(n1,1))*(square(n3,2)-square(n1,2)) - (square(n3,1)-square(n1,1))*(square(n2,2)-square(n1,2));
		paras_temp(1) = (square(n2,2)-square(n1,2))*(square(n3,0)-square(n1,0)) - (square(n3,2)-square(n1,2))*(square(n2,0)-square(n1,0));
		paras_temp(2) = (square(n2,0)-square(n1,0))*(square(n3,1)-square(n1,1)) - (square(n3,0)-square(n1,0))*(square(n2,1)-square(n1,1));
		paras_temp(3) = -paras_temp(0)*square(n1,0)-paras_temp(1)*square(n1,1)-paras_temp(2)*square(n1,2);

		//calculate error
		float deno = sqrt(paras_temp(0)*paras_temp(0) + paras_temp(1)*paras_temp(1) + paras_temp(2)*paras_temp(2));
		int consensus_num = 0;
		float total_error = 0.f;
		for(int x=0; x<pixel_counter; x++)
		{
			//cout<<"row"<<square.row(x)<<endl;
			dist(x) = fabs(paras_temp(0)*square(x,0) + paras_temp(1)*square(x,1) + paras_temp(2)*square(x,2) + paras_temp(3))/deno;
			if(dist(x) < dist_filter)
			{
				consensus_num ++;
				total_error += dist(x);
			} 
		}
		float average_error = total_error / consensus_num;
		float error = average_error / (sqrt(consensus_num)*ratio);

		//compare model
		if(error < error_min)
		{
			error_min = error;
			paras = paras_temp;
			plane_point = (square.row(n1) + square.row(n2) + square.row(n3)) / 3;
		}
	}
	plane_vec(0) = paras(0); 
	plane_vec(1) = paras(1); 
	plane_vec(2) = paras(2); 
	plane_vec.normalize();

	//cout<<"paras"<<paras<<endl;
	return paras;
}

void drawArrow(cv::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha, cv::Scalar color, int thickness, int lineType)
{
    const double PI = 3.1415926;
    Point arrow;
    //计算 θ 角（最简单的一种情况在下面图示中已经展示，关键在于 atan2 函数，详情见下面）   
    double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));
    line(img, pStart, pEnd, color, thickness, lineType);

    //计算箭角边的另一端的端点位置（上面的还是下面的要看箭头的指向，也就是pStart和pEnd的位置） 
    arrow.x = pEnd.x + len * cos(angle + PI * alpha / 180);
    arrow.y = pEnd.y + len * sin(angle + PI * alpha / 180);
    line(img, pEnd, arrow, color, thickness, lineType);
    arrow.x = pEnd.x + len * cos(angle - PI * alpha / 180);
    arrow.y = pEnd.y + len * sin(angle - PI * alpha / 180);
    line(img, pEnd, arrow, color, thickness, lineType);
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "depth_seg");
	ros::NodeHandle na;

	dep_frame=imread("/home/clarence/Dataset/rgbd_dataset_freiburg3_sitting_static/depth/1341845688.562015.png",CV_LOAD_IMAGE_ANYDEPTH);

	if(!dep_frame.empty()) { 
		//imshow("dep",dep_frame);
		//waitKey();

		square_nw = WIDTH / STEP;
		square_nh = HEIGHT / STEP;
		square_num = square_nw * square_nh;

		//Mat to Eigen Matrix 
		MatrixXf depth_mtr(HEIGHT, WIDTH);

	    for( size_t nrow = 0; nrow < dep_frame.rows; nrow++)  
	    {  
	        for(size_t ncol = 0; ncol < dep_frame.cols; ncol++)  
	        {   
	            depth_mtr(nrow,ncol) = dep_frame.at<ushort>(nrow,ncol);      
	        }  
	    }

	    MatrixXf planes_mtr(square_num, 4); //to store plane parameters
	    MatrixXf planes_vec_mtr(square_num, 3); 
	    MatrixXf central_mtr(square_num, 3); //central point position for each square fitting plane

		for(int i = 0; i < square_nw; i++) //divide squares
		{
			for(int j = 0; j < square_nh; j++)
			{
				int start_x = i*STEP;
				int start_y = j*STEP;
				MatrixXf square(STEP*STEP, 3); //(X,Y,Z) for each pixel
				int pixel_counter = 0; 

				for(int x = start_x; x < start_x+STEP; x++)
				{
					for(int y = start_y; y < start_y+STEP; y++)
					{
						//cout<<"("<<x<<","<<y<<") ";
						if(depth_mtr(y,x) > 500.f)
						{
							square(pixel_counter, 2) = depth_mtr(y,x) / factor;
							square(pixel_counter, 0) = (x - cx) * square(pixel_counter, 2) / fx;
							square(pixel_counter, 1) = (y - cy) * square(pixel_counter, 2) / fy;
							pixel_counter ++;  //only save valid points
						}
					}
				}
				

				if(pixel_counter < VALID_THRE) 
				{
					planes_mtr.row(i*square_nh + j) = Vector4f::Zero();
					central_mtr.row(i*square_nh + j) = Vector3f::Zero();
					planes_vec_mtr.row(i*square_nh + j) = Vector3f::Zero();
				}
				else
				{
					Vector4f paras;
					Vector3f plane_vec;
					Vector3f plane_point;
					planes_mtr.row(i*square_nh + j) = plane_ransac(square, paras, plane_vec, plane_point, pixel_counter);
					planes_vec_mtr.row(i*square_nh + j) = plane_vec;
					//central_mtr.row(i*square_nh + j) = square.row(pixel_counter/2);
					central_mtr.row(i*square_nh + j) = plane_point;
				} 
					
				
			}
		}
		//cout<< planes_vec_mtr;

		/*display*/
		Mat disp_img(HEIGHT,WIDTH,CV_8UC3,Scalar(0,0,0));
		for(int i = 0; i < square_nw; i++) //divide squares
		{
			for(int j = 0; j < square_nh; j++)
			{
				int start_x = i*STEP;
				int start_y = j*STEP;
				float dep_thre_max = 10.0;
				float dep_thre_min = 0.5;
				int color = (int) ((dep_thre_max-central_mtr(i*square_nh+j, 2)) / dep_thre_max  * (255 * 3 -200));
				int red = 0;
				int green = 0;
				int blue = 0;

				if (color != (765 - 200))
				{
					if(color > 360) red = color - 360;
					if(color > 155 && color < 410) blue = 255 - (color - 155);
					if(color > 0 && color < 255) green = 255 - color;
				}
				
				rectangle(disp_img, Point(start_x, start_y), Point(start_x+STEP, start_y+STEP), Scalar(blue, green, red), -1, 8);
				int arrow_x = (int)(planes_vec_mtr(i*square_nh+j, 0) * STEP / 2);
				int arrow_y = (int)(planes_vec_mtr(i*square_nh+j, 1) * STEP / 2);
				drawArrow(disp_img, Point(start_x + STEP/2, start_y + STEP/2), Point(start_x + STEP/2 + arrow_x, start_y + STEP/2 + arrow_y), 2, 30, Scalar(200, 50, 0),1, 4);
			}
		}

		imshow("disp_img",disp_img);
		waitKey();

		/*depth image process*/
		/*MatrixXf cluster_mtr = central_mtr;
		for(int i = 1; i < square_nw; i++) //divide squares
		{
			for(int j = 1; j < square_nh; j++)
			{

			}
		}*/

	} 
	else
		cout<<"Wrong Image!"<<endl;


	return 0;
}

