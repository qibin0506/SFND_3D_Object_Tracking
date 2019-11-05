# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

## Rubric

### FP.1 : Match 3D Objects
The matchBoundingBoxes method,
``` c++
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
```

### FP.2 : Compute Lidar-based TTC
I computed the ttc of lidar by using `x * dt / (prev_x - x)`

```c++
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
```

### FP.3 : Associate Keypoint Correspondences with Bounding Boxes
``` c++
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
```

### FP.4 : Compute Camera-based TTC
``` c++
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

```

### FP.5 : Performance Evaluation 1
TTC from Lidar is not correct because of some outliers and some unstable points from preceding vehicle's front mirrors, those need to be filtered out . Here we adapt a bigger shrinkFactor = 0.2, to get more reliable and stable lidar points. Then get a more accurate results.

### FP.6 : Performance Evaluation 2
The TOP3 detector/descriptor combinations as the best choice for our purpose of detecting keypoints on vehicles are: 
SHITOMASI/BRISK

SHITOMASI/BRIEF

SHITOMASI/ORB
