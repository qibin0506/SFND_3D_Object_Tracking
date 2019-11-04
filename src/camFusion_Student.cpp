
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    double dist_mean = 0;
    std::vector<cv::DMatch> roi;

    for (auto it = kptMatches.begin(); it != kptMatches.end(); ++it) {
        cv::KeyPoint kp = kptsCurr.at(it->trainIdx);
        if (boundingBox.roi.contains(cv::Point(kp.pt.x, kp.pt.y))) {
            roi.push_back(*it);
        }
    }

    for (auto it = roi.begin(); it != roi.end(); ++it) {
        dist_mean += it->distance;
    }

    if (roi.size() > 0) {
        dist_mean /= roi.size();
    } else {
        return;
    }

    double threshold = dist_mean * 0.7;
    for (auto it = roi.begin(); it != roi.end(); ++it) {
        if (it->distance < threshold) {
            boundingBox.kptMatches.push_back(*it);
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{   
    double dt = 1 / frameRate;
    std::vector<double> dist_radios;

    for (auto it = kptMatches.begin(); it != kptMatches.end() - 1; ++it) {
        cv::KeyPoint cur_outer_kp = kptsCurr.at(it->trainIdx);
        cv::KeyPoint prev_outer_kp = kptsPrev.at(it->queryIdx);

        for (auto it1 = kptMatches.begin() + 1; it1 != kptMatches.end(); ++it1) {
            double min_dist = 100.0;
            cv::KeyPoint cur_inner_kp = kptsCurr.at(it1->trainIdx);
            cv::KeyPoint prev_inner_kp = kptsPrev.at(it1->queryIdx);

            double cur_dist = cv::norm(cur_outer_kp.pt - cur_inner_kp.pt);
            double prev_dist = cv::norm(prev_outer_kp.pt - prev_inner_kp.pt);
            if (prev_dist > std::numeric_limits<double>::epsilon() && cur_dist >= min_dist) {
                double dist_radio = cur_dist / prev_dist;
                dist_radios.push_back(dist_radio);
            }
        }
    }

    if (dist_radios.size() == 0) {
        TTC = NAN;
        return;
    }

    std::sort(dist_radios.begin(), dist_radios.end());
    long med_index = floor(dist_radios.size() / 2.0);
    double med_dist_radio = dist_radios.size() % 2 == 0 ? (dist_radios[med_index - 1] + dist_radios[med_index]) / 2.0 : dist_radios[med_index];
    TTC = -dt / (1 - med_dist_radio);
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double dt = 1 / frameRate;
    double lane_width = 4;
    vector<double> prev_x, cur_x;

    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it) {
        if (abs(it->y) <= lane_width / 2) {
            prev_x.push_back(it->x);
        }
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it) {
        if (abs(it->y) <= lane_width / 2) {
            cur_x.push_back(it->x);
        }
    }

    double min_prev_x, min_cur_x;
    if (prev_x.size() > 0) {
        for (auto x : prev_x) {
            min_prev_x += x;
        }

        min_prev_x /= prev_x.size();
    }

    if (cur_x.size() > 0) {
        for (auto x : cur_x) {
            min_cur_x += x;
        }

        min_cur_x /= cur_x.size();
    }

    TTC = min_cur_x * dt / (min_prev_x - min_cur_x);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    int pre_bbox_size = prevFrame.boundingBoxes.size();
    int cur_bbox_size = currFrame.boundingBoxes.size();
    int pt_counts[pre_bbox_size][cur_bbox_size] = {};

    for (auto it = matches.begin(); it != matches.end() - 1; ++it) {
        cv::KeyPoint query = prevFrame.keypoints[it->queryIdx];
        auto query_pt = cv::Point(query.pt.x, query.pt.y);
        auto query_found = false;

        cv::KeyPoint train = currFrame.keypoints[it->trainIdx];
        auto train_pt = cv::Point(train.pt.x, train.pt.y);
        auto train_found = false;

        std::vector<int> query_ids, train_ids;
        for (size_t i = 0; i < pre_bbox_size; i++) {
            if (prevFrame.boundingBoxes[i].roi.contains(query_pt)) {
                query_found = true;
                query_ids.push_back(i);
            }
        }

        for (size_t i = 0; i < cur_bbox_size; i++) {
            if (currFrame.boundingBoxes[i].roi.contains(train_pt)) {
                train_found = true;
                train_ids.push_back(i);
            }
        }

        if (query_found && train_found) {
            for (size_t pre_id : query_ids) {
                for (size_t cur_id : train_ids) {
                    pt_counts[pre_id][cur_id] += 1;
                }
            }
        }
    }

    for (size_t i = 0; i < pre_bbox_size; i++) {
            int max_count = 0;
            int max_id = 0;

            for (size_t j = 0; j < cur_bbox_size; j++) {
                if (pt_counts[i][j] > max_count) {
                    max_count = pt_counts[i][j];
                    max_id = j;
                }
            }

            bbBestMatches[i] = max_id;
        }

    for (size_t i = 0; i < pre_bbox_size; i++) {
        cout << "box " << i << "matches box " + bbBestMatches[i] << endl;
    }
}
