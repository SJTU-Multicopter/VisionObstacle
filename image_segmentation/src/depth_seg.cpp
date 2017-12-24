#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <math.h>
#include <Eigen/Core>  
#include <Eigen/Dense>
#include <Eigen/Geometry>  
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <px4_autonomy/Velocity.h>
#include <px4_autonomy/Position.h> 
#include <std_msgs/Float32.h>

using namespace std;
using namespace cv;
using namespace Eigen;

#define WIDTH 640
#define HEIGHT 360
#define STEP 5
#define VALID_THRE 10

Mat dep_frame(WIDTH,HEIGHT,CV_16UC1);
MatrixXf depth_mtr(HEIGHT, WIDTH);

int square_nw, square_nh, square_num;
int flag = 0;
//sensor_msgs::ImagePtr msg;

//camera parameters
float fx = 205.47;
float fy = 205.47;
float cx = 320.5;
float cy = 180.5;
float factor = 5000.0;

Vector3f fly_direction;
Vector3f control_direction;
Vector3f current_position;
Vector3f current_posture; //yaw, roll, pitch
std_msgs::Float32 obstacle_dist_result;

Vector4f &plane_ransac(MatrixXf &square, Vector4f &paras, Vector3f &plane_vec, Vector3f &plane_point, int &pixel_counter, int iterations = 5, float dist_filter = 0.2f, float ratio = 1.f) //Use RANSAC to fit planes. MatrixXf square(STEP*STEP, 3), (X,Y,Z) for each pixel
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


void chatterCallback_depth(const sensor_msgs::PointCloud2& cloud_msg)
{
    //pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2;
   // pcl_conversions::toPCL(*cloud_msg, *cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pcl(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(cloud_msg, *cloud_pcl);

    for(int i = 0; i < HEIGHT; i++)
    {
        for(int j=0; j< WIDTH; j++)
        {
            depth_mtr(i,j) = (float)cloud_pcl->points[i*WIDTH + j].z * factor;
        }
    }
    
    flag = 1;
    // for(int i = 0; i < 640*360; i++)
    //     cout<<cloud_pcl->points[i].z<<" ";
    //cout<<"*********" << endl;
}

//  void chatterCallback_depth(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input){

//     pcl::PCLPointCloud2 pcl_pc2;
//     pcl_conversions::toPCL(*input,pcl_pc2);
//     pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);
//     //do stuff with temp_cloud here
//     cout<<pcl_pc2->points[0]<<endl;
// }

void chatterCallback_body(const px4_autonomy::Velocity& msg)
{
    fly_direction(0) = msg.x;
    fly_direction(1) = msg.y;
    fly_direction(2) = msg.z;
}

void chatterCallback_ps2(const px4_autonomy::Velocity& msg)
{
    control_direction(0) = msg.x;
    control_direction(1) = msg.y;
    control_direction(2) = msg.z;
}

void chatterCallback_pose(const px4_autonomy::Position& msg)
{
    current_position(0) = msg.x;
    current_position(1) = msg.y;
    current_position(2) = msg.z;
    current_posture(0) = msg.yaw;
    current_posture(1) = msg.roll;
    current_posture(2) = msg.pitch;
}

float transvection(Vector3f &x, Vector3f &y)
{
    return x(0)*y(0) + x(1)*y(1) + x(2)*y(2);
}

float vector_length(Vector3f &x)
{
    return sqrt(x(0)*x(0) + x(1)*x(1) + x(2)*x(2));
}

float acos_near(float x)
{
    return 1.5708 * (1.0 - x);
}

float angle_between_vectors(Vector3f &x, Vector3f &y)
{
    //return acos(transvection(x,y) / vector_length(x) / vector_length(y));
    return acos_near(transvection(x,y) / vector_length(x) / vector_length(y));
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "depth_seg");
    ros::NodeHandle nh;

    ros::Subscriber depth_sub = nh.subscribe("/camera/depth/points", 1, chatterCallback_depth);
    ros::Subscriber local_pose_sub = nh.subscribe("/px4/pose", 1,chatterCallback_pose);
    ros::Subscriber body_vel_sub = nh.subscribe("/px4/body_vel", 1,chatterCallback_body);
    ros::Subscriber ps2_vel_sub = nh.subscribe("/px4/ps2_vel", 1,chatterCallback_ps2);

    ros::Publisher obstacle_dist_pub = nh.advertise<std_msgs::Float32>("/obstacle_dist", 1); 

    //dep_frame=imread("/home/clarence/Dataset/rgbd_dataset_freiburg3_sitting_static/depth/1341845688.562015.png",CV_LOAD_IMAGE_ANYDEPTH);

    square_nw = WIDTH / STEP;
    square_nh = HEIGHT / STEP;
    square_num = square_nw * square_nh;
    int slade_square_thred = square_nh * 0.2; 
    int body_square_min = square_nw * 0.15;
    int body_square_max = square_nw - body_square_min;

    ros::Rate loop_rate(20);
    while(nh.ok())
    {
       // if(!dep_frame.empty()) {
        if(flag > 0) {    
            //imshow("dep",dep_frame);
           // waitKey();

            

            //Mat to Eigen Matrix

            /*for( size_t nrow = 0; nrow < dep_frame.rows; nrow++)
            {
                for(size_t ncol = 0; ncol < dep_frame.cols; ncol++)
                {
                    depth_mtr(nrow,ncol) = dep_frame.at<ushort>(nrow,ncol);
                }
            }*/

            MatrixXf planes_mtr(square_num, 4); //to store plane parameters
            MatrixXf planes_vec_mtr(square_num, 3); 
            MatrixXf central_mtr(square_num, 3); //central point position for each square fitting plane, (x,y,z)=(front, left, up)

            for(int i = 0; i < square_nw; i++) //divide squares
            {
                for(int j = 0; j < square_nh; j++)
                {
                    int start_x = i*STEP;
                    int start_y = j*STEP;
                    MatrixXf square(STEP*STEP, 3); //(X,Y,Z) for each pixel, X is rightside, Y is down side and Z is front side(depth).
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

           

            float min_obstacle_dist = 100.0;
            Vector3f obstacle_direction;
            obstacle_direction(0) = 1;
            obstacle_direction(1) = 0;
            obstacle_direction(2) = 0;

            /*Add pose crrection here*/
            Eigen::Vector3f ea0(current_posture(2), 0.0, current_posture(1));  
            Eigen::Matrix3f R;  
            R = Eigen::AngleAxisf(ea0[0], Eigen::Vector3f::UnitX())  
                * Eigen::AngleAxisf(ea0[1], Eigen::Vector3f::UnitY())  
                * Eigen::AngleAxisf(ea0[2], Eigen::Vector3f::UnitZ()); 

            central_mtr = (R * central_mtr.transpose()).transpose();
            planes_vec_mtr = (R * planes_vec_mtr.transpose()).transpose();
            /*for(int i = 0; i < square_nw * square_nh; i++)
            {
                Vector3f central_vec = central_mtr.row(i);
                Vector3f plane_vec = planes_vec_mtr.row(i);

                central_vec = R * central_vec;
                plane_vec = R * plane_vec;

                central_mtr.row(i) = central_vec;
                planes_vec_mtr.row(i) = plane_vec;
            }*/

            //central_mtr = central_mtr * R;
            //planes_vec_mtr = planes_vec_mtr * R;

            /*************************/

             /*For test*/
            // cout<<central_mtr.row(320)<<endl<<"****"<<endl;
            // cout<<central_mtr.row(321)<<endl<<"****"<<endl;
            // central_mtr(320, 2) = 5.0;
            // central_mtr(321, 2) = 2.0;

            /****/

            /**Obstcle distance calculation**/

            if(fly_direction(0) < -0.1)
            {
                min_obstacle_dist = 100;
            }
            else
            {
                for(int i = 0; i < square_num; i++)
                {
                    central_mtr(i, 1) = central_mtr(i, 1) * 3;
                    Vector3f direction;
                    direction(0) = central_mtr(i,2);
                    direction(1) = -central_mtr(i,0);
                    direction(2) = -central_mtr(i,1);

                    if(central_mtr(i, 2) > 1.5 && -central_mtr(i, 1) > -current_position(2) + 0.3) //do not count ground
                    {
                        
                        float theta0 = angle_between_vectors(fly_direction, control_direction);
                        Vector3f middle_direction = (fly_direction + control_direction) / 2.f;

                        float delt_theta = 0.5;
                        if(angle_between_vectors(fly_direction, direction) + angle_between_vectors(control_direction, direction) < theta0 + delt_theta 
                            && transvection(middle_direction, direction) > 0) //obstacle judge
                        {
                            float dist = vector_length(direction);
                            if(min_obstacle_dist >  dist && dist > 0.3)
                            {
                                min_obstacle_dist = dist;
                                obstacle_direction = direction;
                            }
                                
                        }
                    }
                    else if(central_mtr(i, 2) > 0.6 && -central_mtr(i, 1) > -current_position(2) + 0.3)//do not count ground and slades 
                    {
                        float dist = vector_length(direction);
                        if(min_obstacle_dist >  dist && dist > 0.5)
                        {
                            min_obstacle_dist = dist;
                        }
                    }
                    else if(central_mtr(i, 2) > 0.1 && -central_mtr(i, 1) > -current_position(2) + 0.3 
                            && i%square_nh > slade_square_thred && i%square_nw > body_square_min && i%square_nw < body_square_max) //handle slades and body
                    {
                        min_obstacle_dist = 0.5;
                    }

                }
            }

            
            //cout<<obstacle_direction<<"**"<<endl;
            obstacle_dist_result.data = min_obstacle_dist;
            obstacle_dist_pub.publish(obstacle_dist_result);


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
            waitKey(40);
        }
        else
            cout<<"Wrong Image!"<<endl;

        ros::spinOnce();
        loop_rate.sleep();
    }

	return 0;
}

