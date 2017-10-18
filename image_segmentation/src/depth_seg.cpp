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
#define STEP 20
#define VALID_THRE 50

Mat dep_frame(WIDTH,HEIGHT,CV_16UC1);

int square_nw, square_nh, square_num;
//sensor_msgs::ImagePtr msg;

//camera parameters
float fx = 525.0;
float fy = 525.0;
float cx = 319.5;
float cy = 239.5;
float factor = 5000.0;

Vector4f &plane_ransac(MatrixXf &square, Vector4f &paras, Vector3f &plane_vec, int &pixel_counter, int iterations = 10, float dist_filter = 0.2f, float ratio = 1.f) //Use RANSAC to fit planes. MatrixXf square(STEP*STEP, 3), (X,Y,Z) for each pixel
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
		}
	}
	plane_vec(0) = paras(0); 
	plane_vec(1) = paras(1); 
	plane_vec(2) = paras(2); 
	plane_vec.normalize();

	//cout<<"paras"<<paras<<endl;
	return paras;
}



int main(int argc, char** argv)
{
	ros::init(argc, argv, "depth_seg");
	ros::NodeHandle na;

	dep_frame=imread("/home/clarence/Dataset/rgbd_dataset_freiburg3_sitting_static/depth/1341845688.562015.png",CV_LOAD_IMAGE_ANYDEPTH);

	if(!dep_frame.empty()) { 
		imshow("dep",dep_frame);
		waitKey();

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
	    MatrixXf central_mtr(square_num, 3);

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
					planes_mtr.row(i*square_nh + j) = plane_ransac(square, paras, plane_vec, pixel_counter);
					planes_vec_mtr.row(i*square_nh + j) = plane_vec;
					central_mtr.row(i*square_nh + j) = square.row(pixel_counter/2);
				} 
					
				
			}
		}
		cout<< planes_vec_mtr;
	} 
	else
		cout<<"Wrong Image!"<<endl;


	return 0;
}

