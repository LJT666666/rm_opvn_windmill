//
// Created by ljt666666 on 22-8-7.
//
#include "rm_opvn_windmill/opvn_windmill.h"
#include <pluginlib/class_list_macros.h>
#include "rm_vision/vision_base/processor_interface.h"
#include <ros/package.h>
#include <ros/callback_queue.h>

using namespace InferenceEngine;
using namespace cv;

PLUGINLIB_EXPORT_CLASS(rm_opvn_windmill::OpvnProcessor, nodelet::Nodelet)

namespace rm_opvn_windmill {
    void OpvnProcessor::onInit() {
        ros::NodeHandle& nh = getPrivateNodeHandle();
        static ros::CallbackQueue my_queue;
        nh.setCallbackQueue(&my_queue);
        initialize(nh);
        my_thread_ = std::thread([](){
            ros::SingleThreadedSpinner spinner;
            spinner.spin(&my_queue);
        });
    }

    void OpvnProcessor::initialize(ros::NodeHandle &nh) {
        nh_ = ros::NodeHandle(nh, "rm_opvn_windmill");
        if(!nh.getParam("rotate", rotate_))
            ROS_WARN("No rotate specified");
        if(!nh.getParam("twelve_classes", twelve_classes_))
            ROS_WARN("No twelve_classes specified");
        if(!nh.getParam("xml_path", xml_path_))
            ROS_WARN("No xml_path specified");
        if(!nh.getParam("bin_path", bin_path_))
            ROS_WARN("No bin_path specified");
        if(!nh.getParam("cof_threshold", cof_threshold_))
            ROS_WARN("No cof_threshold specified");
        if(!nh.getParam("nms_area_threshold", nms_area_threshold_))
            ROS_WARN("No xml_path specified");
        if(!nh.getParam("class_num", class_num_))
            ROS_WARN("No class_num_ specified");
        if(!nh.getParam("input_row", input_row_))
            ROS_WARN("No input_row specified");
        if(!nh.getParam("input_col", input_col_))
            ROS_WARN("No input_col specified");

        xml_path_ = ros::package::getPath("rm_opvn_windmill") + xml_path_;
        bin_path_ = ros::package::getPath("rm_opvn_windmill") + bin_path_;
        ROS_INFO("model_path:%s", (xml_path_+bin_path_).c_str());

        //Initialize the inference parameters
        Core ie;
        CNNNetwork network = ie.ReadNetwork(xml_path_, bin_path_);
        if (network.getOutputsInfo().size() != 1)
            throw std::logic_error("Sample supports topologies with 1 output only");
        if (network.getInputsInfo().size() != 1)
            throw std::logic_error("Sample supports topologies with 1 input only");
        InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
        input_name_ = network.getInputsInfo().begin()->first;
        output_info_ = network.getOutputsInfo().begin()->second;
        output_name_ = network.getOutputsInfo().begin()->first;
        output_info_->setPrecision(Precision::FP32);
        executable_network_ = ie.LoadNetwork(network, "CPU");
        infer_request_ = executable_network_.CreateInferRequest();

        opvn_cfg_srv_ = new dynamic_reconfigure::Server<rm_opvn_windmill::OpvnConfig>(ros::NodeHandle(nh_, "opvn_condition"));
        opvn_cfg_cb_ = boost::bind(&OpvnProcessor::opvnconfigCB, this, _1, _2);
        opvn_cfg_srv_->setCallback(opvn_cfg_cb_);

        it_ = make_shared<image_transport::ImageTransport>(nh_);
        debug_pub_ = it_->advertise("debug_image", 1);
        cam_sub_ = it_->subscribeCamera("/galaxy_camera/image_raw", 1, &OpvnProcessor::callback, this);
//        bag_sub_ = it_->subscribe("/hk_camera/image_raw", 1, &OpvnProcessor::callback, this);

        target_pub_ = nh.advertise<decltype(target_array_)>("/processor/result_msg", 1);
//        TalkMsg_pub_ = nh.advertise<std_msgs::Float32>("wind_speed", 1);
    }

    void OpvnProcessor::opvnconfigCB(rm_opvn_windmill::OpvnConfig& config, uint32_t level)
    {
        cof_threshold_ = config.cof_threshold;
        nms_area_threshold_ = config.nms_area_threshold;
        rotate_ = config.rotate;
        twelve_classes_ = config.twelve_classes;
//        target_type_ = config.target_color;
    }

    void OpvnProcessor::imageProcess(cv_bridge::CvImagePtr &cv_image) {
        if (cv_image->image.empty()) {
            ROS_ERROR("invalid image input");
            return;
        }
        cv_image->image.copyTo(image_raw_);
    }

    void OpvnProcessor::findArmor() {
        InferRequest::Ptr infer = executable_network_.CreateInferRequestPtr();
//        image_raw_ = imread("/home/ljt666666/catkin_ws/src/rm_visplugin/rm_opvn_windmill/images/blue_windmill.png");
        if(rotate_)
            cv::rotate(image_raw_, image_raw_, cv::ROTATE_180);
        square_image_ = staticResize(image_raw_);
        Blob::Ptr img_blob = infer_request_.GetBlob(input_name_);     // just wrap Mat data by Blob::Ptr
        blobFromImage(square_image_, img_blob);
        //infer
        infer_request_.Infer();
        // get infer result
        parseModel();
    }

    void OpvnProcessor::parseModel() {
        const Blob::Ptr output_blob = infer_request_.GetBlob(output_name_);
        MemoryBlob::CPtr moutput = as<MemoryBlob>(output_blob);
        if (!moutput) {
            throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                                   "but by fact we were not able to cast output to MemoryBlob");
        }
        // locked memory holder should be alive all time while access to its buffer
        // happens
        auto moutput_holder = moutput->rmap();
        const float *net_pred = moutput_holder.as<const PrecisionTrait<Precision::FP32>::value_type *>();
        decodeOutputs(net_pred);
    }

    void OpvnProcessor::decodeOutputs(const float *net_pred) {
        std::vector<Target> proposals;
        std::vector<int> strides = {8, 16, 32};
        std::vector<GridAndStride> grid_strides;
        std::vector<int> picked;

        generateGridsAndStride(input_col_, input_row_, strides, grid_strides);
        generateYoloxProposals(grid_strides, net_pred, proposals);
        //ROS_INFO("proposals size is : %d", proposals.size());
        qsortDescentInplace(proposals, proposals.size() - 1);
        nmsSortedBoxes(proposals, picked, nms_area_threshold_);

        objects_ = proposals;
        int count = proposals.size();

//        target_array_.detections.clear();
//        if(count == 0){
//            rm_msgs::TargetDetection one_target;
//            one_target.id = 0;
//            target_array_.detections.emplace_back(one_target);
//        }

        for (int i = 0; i < count; i++) {
            rm_msgs::TargetDetection one_target;
            one_target.confidence = proposals[i].prob;
            one_target.id = 1;
            if(proposals[i].label == 0)
                continue;
            int32_t temp[10];
            for (int k = 0; k < 5; k++)
            {
                temp[k * 2] = static_cast<int32_t>(roundf(proposals[i].points[k * 2])/r_);
                temp[k * 2 + 1] = static_cast<int32_t>(roundf(proposals[i].points[(k * 2)+1]/r_));
            }
            memcpy(&one_target.pose.position.x, &temp[4], sizeof(int32_t) * 2); //R icon
            memcpy(&one_target.pose.orientation.x, &temp[0], sizeof(int32_t) * 2);
            memcpy(&one_target.pose.orientation.y, &temp[2], sizeof(int32_t) * 2);
            memcpy(&one_target.pose.orientation.z, &temp[6], sizeof(int32_t) * 2);
            memcpy(&one_target.pose.orientation.w, &temp[8], sizeof(int32_t) * 2);
            target_array_.detections.emplace_back(one_target);

            double length = pow(pow(proposals[i].points[0]/r_ - proposals[i].points[2]/r_, 2) + pow(proposals[i].points[1]/r_ - proposals[i].points[3]/r_, 2), 0.5);
            double width = pow(pow(proposals[i].points[2]/r_ - proposals[i].points[4]/r_, 2) + pow(proposals[i].points[3]/r_ - proposals[i].points[5]/r_, 2), 0.5);
            double ratio = length / width;
            //ROS_INFO("length:%f, width: %fratio:%f", length, width, ratio);
//            int32_t data[4 * 2];
//            memcpy(&data[0], &one_target.pose.position.x, sizeof(int32_t) * 2);
//            memcpy(&data[2], &one_target.pose.orientation.y, sizeof(int32_t) * 2);
//            memcpy(&data[4], &one_target.pose.orientation.z, sizeof(int32_t) * 2);
//            memcpy(&data[6], &one_target.pose.orientation.w, sizeof(int32_t) * 2);
//            for (int k = 1; i < k; i++){
//                ROS_INFO("%d:%d", i, data[i]);
//            }
        }
    }

    void OpvnProcessor::draw() {
        //static const char *class_names[] = {"red_2", "red_3", "red_4", "red_5"};
        for(auto & object : objects_){
            line(image_raw_, Point(object.points[0]/r_, object.points[1]/r_), Point(object.points[2]/r_, object.points[3]/r_), Scalar(0, 255, 0), 2, 4);
            line(image_raw_, Point(object.points[2]/r_, object.points[3]/r_), Point(object.points[4]/r_, object.points[5]/r_), Scalar(0, 255, 0), 2, 4);
            line(image_raw_, Point(object.points[4]/r_, object.points[5]/r_), Point(object.points[6]/r_, object.points[7]/r_), Scalar(0, 255, 0), 2, 4);
            line(image_raw_, Point(object.points[6]/r_, object.points[7]/r_), Point(object.points[8]/r_, object.points[9]/r_), Scalar(0, 255, 0), 2, 4);
            line(image_raw_, Point(object.points[8]/r_, object.points[9]/r_), Point(object.points[0]/r_, object.points[1]/r_), Scalar(0, 255, 0), 2, 4);
//            cv::putText(image_raw_, to_string(object.label), cv::Point(object.points[0]/r_, object.points[3]/r_-40),cv::FONT_HERSHEY_SIMPLEX, 1, Scalar (0, 255, 0), 3);
            cv::putText(image_raw_, to_string(1), cv::Point(object.points[0]/r_, object.points[1]/r_),cv::FONT_HERSHEY_SIMPLEX, 1, Scalar (0, 255, 0), 3);
            cv::putText(image_raw_, to_string(2), cv::Point(object.points[2]/r_, object.points[3]/r_),cv::FONT_HERSHEY_SIMPLEX, 1, Scalar (0, 255, 0), 3);
            cv::putText(image_raw_, to_string(3), cv::Point(object.points[6]/r_, object.points[7]/r_),cv::FONT_HERSHEY_SIMPLEX, 1, Scalar (0, 255, 0), 3);
            cv::putText(image_raw_, to_string(4), cv::Point(object.points[8]/r_, object.points[9]/r_),cv::FONT_HERSHEY_SIMPLEX, 1, Scalar (0, 255, 0), 3);
            //            ROS_INFO("%d", object.)

////            speed_solver
//            if(object.label == 1){
//                update_Fan(object);
//                speed_Solution();
//            }
        }

    }

//    void OpvnProcessor::updateFan(const Target& object) {
//        if(!init_flag_){
//            prev_fan_ = object;
//            init_flag_ = true;
//        }
//        else{
//            last_fan_ = prev_fan_;
//            prev_fan_ = object;
//        }
//    }
//
//    void OpvnProcessor::speedSolution() {
//        float prev_total_x;
//        float prev_total_y;
//        for (int i = 0;i < 2;i++){
//            prev_total_x += prev_fan_.points[2 * i];
//            prev_total_y += prev_fan_.points[2 * i + 1];
//        }
//        for (int i = 0;i < 2;i++){
//            prev_total_x += prev_fan_.points[2 * i + 6];
//            prev_total_y += prev_fan_.points[2 * i + 6 + 1];
//        }
//
//        prev_fan_.armor_center_points.x = prev_total_x / 4;
//        prev_fan_.armor_center_points.y = prev_total_y / 4;
//        prev_fan_.r_points.x = prev_fan_.points[4];
//        prev_fan_.r_points.y = prev_fan_.points[5];
//        if(!init_flag_){
//            delat_t_ = target_array_.header.stamp;
//            return;
//        }
//        float angle = linesOrientation(prev_fan_.armor_center_points, prev_fan_.r_points, last_fan_.armor_center_points, last_fan_.r_points, 0);
//        float delat_t = (target_array_.header.stamp - delat_t_).toSec();
//        delat_t_ = target_array_.header.stamp;
//        wind_speed_ = angle / delat_t;
//        ROS_INFO("%f rad/s", wind_speed_*0.25);
//        std_msgs::Float32 msg;
//        msg.data = wind_speed_*0.25;
////        TalkMsg_pub_.publish(msg);
//    }
//
////根据两直线上两点计算两直线夹角
////当flag=0时返回弧度，当flag!=0时返回角度
//    float OpvnProcessor::linesOrientation(const cv::Point2f& A1, const cv::Point2f& A2, const cv::Point2f& B1, const cv::Point2f& B2, int flag)
//    {
//        //【1】根据直线上两点计算斜率
//        float k_line1 = (A2.y - A1.y)/(A2.x - A1.x);
//        float k_line2 = (B2.y - B1.y) / (B2.x - B1.x);
//        if(B2.x == B1.x)
//            return 0;
//        //【2】根据两个斜率计算夹角
//        float tan_k;                            //直线夹角正切值
//        tan_k = (k_line2 - k_line1) / (1 + k_line2*k_line1);
//        float lines_arctan = atan(tan_k);            //反正切获取夹角
////        ROS_INFO("kline1:%f", k_line1);
////        ROS_INFO("kline2:%f", k_line2);
////        ROS_INFO("tan_k:%f", tan_k);
//        //【3】以角度或弧度形式返回夹角
//        if (flag == 0)
//        {
//            return lines_arctan;
//        }
//        else
//        {
//            return lines_arctan* 180.0 / 3.1415926;
//        }
//    }

    Mat OpvnProcessor::staticResize(cv::Mat &img) {
        r_ = std::min(input_col_ / (img.cols * 1.0), input_row_ / (img.rows * 1.0));
        int unpad_w = r_ * img.cols;
        int unpad_h = r_ * img.rows;
        cv::Mat re(unpad_h, unpad_w, CV_8UC3);
        cv::resize(img, re, re.size());

        //cv::Mat out(INPUT_W, INPUT_H, CV_8UC3, cv::Scalar(114, 114, 114));
        cv::Mat out(input_row_, input_col_, CV_8UC3, cv::Scalar(114, 114, 114));
        re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
        return out;
    }

    void OpvnProcessor::blobFromImage(cv::Mat &img, Blob::Ptr &blob) {
        int channels = 3;
        int img_h = img.rows;
        int img_w = img.cols;
        InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);

        // locked memory holder should be alive all time while access to its buffer happens
        auto mblob_holder = mblob->wmap();
        float *blob_data = mblob_holder.as<float *>();

        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < img_h; h++) {
                for (size_t w = 0; w < img_w; w++) {
                    blob_data[c * img_w * img_h + h * img_w + w] =
                            (float) img.at<cv::Vec3b>(h, w)[c];
                }
            }
        }
    }

    void OpvnProcessor::generateGridsAndStride(const int target_w, const int target_h, std::vector<int> &strides,
                                               std::vector<GridAndStride> &grid_strides) {
        for (auto stride : strides) {
            int num_grid_w = target_w / stride;
            int num_grid_h = target_h / stride;
            for (int g1 = 0; g1 < num_grid_h; g1++) {
                for (int g0 = 0; g0 < num_grid_w; g0++) {
                    grid_strides.push_back((GridAndStride) {g0, g1, stride});
                }
            }
        }
    }

    void OpvnProcessor::qsortDescentInplace(std::vector<Target> &faceobjects, int right) {
        if (faceobjects.empty())
            return;
        int i = 0;
        int j = right;
        float p = faceobjects[(0 + right) / 2].prob;

        while (i <= j) {
            while (faceobjects[i].prob > p)
                i++;

            while (faceobjects[j].prob < p)
                j--;

            if (i <= j) {
                // swap
                std::swap(faceobjects[i], faceobjects[j]);
                i++;
                j--;
            }
        }
    }

    void OpvnProcessor::generateYoloxProposals(std::vector<GridAndStride> grid_strides, const float *net_pred, std::vector<Target>& proposals){

        const int num_anchors = grid_strides.size();

        for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
            const int grid0 = grid_strides[anchor_idx].grid0;
            const int grid1 = grid_strides[anchor_idx].grid1;
            const int stride = grid_strides[anchor_idx].stride;

            const int basic_pos = anchor_idx * (11 + 2 + 2);

            //polygon_yolox
            float x1 = (net_pred[basic_pos + 0] + grid0) * stride;
            float y1 = (net_pred[basic_pos + 1] + grid1) * stride;
            float x2 = (net_pred[basic_pos + 2] + grid0) * stride;
            float y2 = (net_pred[basic_pos + 3] + grid1) * stride;
            float x3 = (net_pred[basic_pos + 4] + grid0) * stride;
            float y3 = (net_pred[basic_pos + 5] + grid1) * stride;
            float x4 = (net_pred[basic_pos + 6] + grid0) * stride;
            float y4 = (net_pred[basic_pos + 7] + grid1) * stride;
            float x5 = (net_pred[basic_pos + 8] + grid0) * stride;
            float y5 = (net_pred[basic_pos + 9] + grid1) * stride;
            int box_color = argmax(net_pred + basic_pos + 11, 2);
            int box_class = argmax(net_pred + basic_pos + 11 + 2, 2);

            float box_objectness = net_pred[basic_pos + 10];
            float color_conf = (net_pred[basic_pos + 11 + box_color]);
            float cls_conf = (net_pred[basic_pos + 11 + 2 + box_class]);

            float box_prob = box_objectness * color_conf * cls_conf;
            if (box_prob > cof_threshold_) {
                Target obj;
                obj.points.push_back(x1);
                obj.points.push_back(y1);
                obj.points.push_back(x2);
                obj.points.push_back(y2);
                obj.points.push_back(x3);
                obj.points.push_back(y3);
                obj.points.push_back(x4);
                obj.points.push_back(y4);
                obj.points.push_back(x5);
                obj.points.push_back(y5);
                /*obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;*/
                obj.label = box_class;
                obj.prob = box_prob;
                proposals.push_back(obj);
//                cout << "x1_" << x1 << endl;
//                cout << "y1_" << y1 <<endl;
//                cout << "x2_" << x2 << endl;
//                cout << "y2_" << y2 << endl;
//                cout << "x3_" << x3 <<endl;
//                cout << "y3_" << y3 <<endl;
//                cout << "x4_" << x4 << endl;
//                cout << "y4_" << y4 << endl;
//                cout << "x5_" << x5 << endl;
//                cout << "y5_" << y5 << endl;
            } // class loop
        } // point anchor loop
    }

    int OpvnProcessor::argmax(const float *ptr, int len)
    {
        int max_arg = 0;
        for (int i = 1; i < len; i++) {
            if (ptr[i] > ptr[max_arg]) max_arg = i;
        }
        return max_arg;
    }

    void OpvnProcessor::nmsSortedBoxes(std::vector<Target> &faceobjects, std::vector<int> &picked, double nms_threshold)  {
        picked.clear();
        PolygonIou polygon_iou;
        vector<Target> results;
        while (!faceobjects.empty())
        {
            results.push_back(faceobjects[0]);
            int index = 1;
            while (index < faceobjects.size()) {
                Target a = faceobjects[0];
                Target b = faceobjects[index];
//                double p[8] = {a.points[0], a.points[1], a.points[6], a.points[7], a.points[4], a.points[5], a.points[2], a.points[3]};
//                double q[8] = {b.points[0], b.points[1], b.points[6], b.points[7], b.points[4], b.points[5], b.points[2], b.points[3]};
//                vector<double> pp, qq;
//                for(int i = 0;i < 8; i++)
//                {
//                    pp.push_back(p[i]);
//                    qq.push_back(q[i]);
//                }
                double p[10] = {a.points[0], a.points[1], a.points[2], a.points[3], a.points[4], a.points[5], a.points[6], a.points[7], a.points[8], a.points[9]};
                double q[10] = {b.points[0], b.points[1], b.points[2], b.points[3], b.points[4], b.points[5], b.points[6], b.points[7], b.points[8], b.points[9]};
                vector<double> pp, qq;
                for(int i = 0;i < 10; i++)
                {
                    pp.push_back(p[i]);
                    qq.push_back(q[i]);
                }
                float iou_value = polygon_iou.iouPoly(pp, qq);
                if (iou_value == 1)
                    iou_value = 0;
                //cout << "iou_value=" << iou_value << endl;
                if (iou_value > nms_threshold) {
                    faceobjects.erase(faceobjects.begin() + index);
                }
                else {
                    index++;
                }
            }
            faceobjects.erase(faceobjects.begin());
        }
        faceobjects = results;
    }

    double PolygonIou::cross(Points o,Points a,Points b){  //Dot multiply
        return(a.x-o.x)*(b.y-o.y)-(b.x-o.x)*(a.y-o.y);
    }

    double PolygonIou::area(Points* ps,int n){
        ps[n]=ps[0];
        double res=0;
        for(int i=0;i<n;i++){
            res+=ps[i].x*ps[i+1].y-ps[i].y*ps[i+1].x;
        }
        return res/2.0;
    }

    int PolygonIou::lineCross(Points a,Points b,Points c,Points d,Points&p){
        double s1,s2;
        s1=cross(c,a,b);
        s2=cross(a,b,d);
        if(sig(s1)==0&&sig(s2)==0) return 2;
        if(sig(s2-s1)==0) return 0;
        p.x=(c.x*s2-d.x*s1)/(s2-s1);
        p.y=(c.y*s2-d.y*s1)/(s2-s1);
        return 1;
    }

    /*
    Polygon cutting
    Cut the polygon p with a straight line ab, after cutting on the left side of the vector (a, b), and save the cut result in situ
    If it degenerates to a point, it also returns, at which point n is 1
     */
    //void polygonCut(Point*p,int&n,Point a,Point b){
    //    static Point pp[maxn];
    //    int m=0;p[n]=p[0];
    //    for(int i=0;i<n;i++){
    //        if(sig(cross(a,b,p[i]))>0) pp[m++]=p[i];
    //        if(sig(cross(a,b,p[i]))!=sig(cross(a,b,p[i+1])))
    //            lineCross(a,b,p[i],p[i+1],pp[m++]);
    //    }
    //    n=0;
    //    for(int i=0;i<m;i++)
    //        if(!i||!(pp[i]==pp[i-1]))
    //            p[n++]=pp[i];
    //    while(n>1&&p[n-1]==p[0])n--;
    //}

    void PolygonIou::polygonCut(Points*p,int&n,Points a,Points b, Points* pp){
        //static Points pp[maxn];
        int m=0;p[n]=p[0];
        for(int i=0;i<n;i++){
            if(sig(cross(a,b,p[i]))>0) pp[m++]=p[i];
            if(sig(cross(a,b,p[i]))!=sig(cross(a,b,p[i+1])))
                lineCross(a,b,p[i],p[i+1],pp[m++]);
        }
        n=0;
        for(int i=0;i<m;i++)
            if(!i||!(pp[i]==pp[i-1]))
                p[n++]=pp[i];
        while(n>1&&p[n-1]==p[0])n--;
    }

    //Returns the directed intersection area of the triangle oab and the triangle oad, with o being the origin
    double PolygonIou::intersectArea(Points a,Points b,Points c,Points d){
        Points o(0,0);
        int s1=sig(cross(o,a,b));
        int s2=sig(cross(o,c,d));
        if(s1==0||s2==0)return 0.0;//退化，面积为0
        if(s1==-1) swap(a,b);
        if(s2==-1) swap(c,d);
        Points p[10]={o,a,b};
        int n=3;
        Points pp[maxn];
        polygonCut(p,n,o,c, pp);
        polygonCut(p,n,c,d, pp);
        polygonCut(p,n,d,o, pp);
        double res=fabs(area(p,n));
        if(s1*s2==-1) res=-res;return res;
    }

    //Finds the intersection area of the two polygons
    double PolygonIou::intersectArea(Points*ps1,int n1,Points*ps2,int n2){
        if(area(ps1,n1)<0) reverse(ps1,ps1+n1);
        if(area(ps2,n2)<0) reverse(ps2,ps2+n2);
        ps1[n1]=ps1[0];
        ps2[n2]=ps2[0];
        double res=0;
        for(int i=0;i<n1;i++){
            for(int j=0;j<n2;j++){
                res+=intersectArea(ps1[i],ps1[i+1],ps2[j],ps2[j+1]);
            }
        }
        return res;//assumeresispositive!
    }

    double PolygonIou::iouPoly(vector<double> p, vector<double> q) {
        Points ps1[maxn],ps2[maxn];
//        int n1 = 4;
//        int n2 = 4;
//        for (int i = 0; i < 4; i++) {
//            ps1[i].x = p[i * 2];
//            ps1[i].y = p[i * 2 + 1];
//            ps2[i].x = q[i * 2];
//            ps2[i].y = q[i * 2 + 1];
//        }
        int n1 = 5;
        int n2 = 5;
        for (int i = 0; i < 5; i++) {
            ps1[i].x = p[i * 2];
            ps1[i].y = p[i * 2 + 1];
            ps2[i].x = q[i * 2];
            ps2[i].y = q[i * 2 + 1];
        }
        double inter_area = intersectArea(ps1, n1, ps2, n2);
        double union_area = fabs(area(ps1, n1)) + fabs(area(ps2, n2)) - inter_area;
        double iou = inter_area / union_area;

        //    cout << "inter_area:" << inter_area << endl;
        //    cout << "union_area:" << union_area << endl;
        //    cout << "iou:" << iou << endl;

        return iou;
    }

    void OpvnProcessor::paramReconfig() {

    }

    void OpvnProcessor::putObj() {

    }

    OpvnProcessor::Object OpvnProcessor::getObj() {

    }

}  // namespace rm_opvn_windmill