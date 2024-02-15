#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {


    Mat image1 = imread("pano1.JPG", 0);
    Mat image2 = imread("pano2.JPG", 0);
    Mat image3 = imread("pano3.JPG", 0);
    Mat image4 = imread("pano4.JPG", 0);

    Mat image1_save = imread("pano1.JPG");
    Mat image2_save = imread("pano2.JPG");
    Mat image3_save = imread("pano3.JPG");
    Mat image4_save = imread("pano4.JPG");
    Mat image123_save;

    Ptr<ORB> orb = ORB::create();
    vector<KeyPoint> keypoints1, keypoints2, keypoints3, keypoints4;
    vector<KeyPoint> keypoints12, keypoints23, keypoints34, keypoints123, keypoints234;
    Mat detectp1, detectp2, detectp3, detectp4, detectp12, detectp23, detectp34, detectp123, detectp234;
    orb->detectAndCompute(image1, Mat(), keypoints1, detectp1);
    orb->detectAndCompute(image2, Mat(), keypoints2, detectp2);
    orb->detectAndCompute(image3, Mat(), keypoints3, detectp3);
    orb->detectAndCompute(image4, Mat(), keypoints4, detectp4);

    BFMatcher matcher(NORM_HAMMING);
    vector<vector<DMatch>> matchp12, matchp23, matchp34, matchp1234, matchp123, matchp234;
    matcher.knnMatch(detectp1, detectp2, matchp12, 2);
    matcher.knnMatch(detectp2, detectp3, matchp23, 2);
    matcher.knnMatch(detectp3, detectp4, matchp34, 2);


    vector<DMatch> goodmatchp;
    vector<Point2f>points1, points2;

    //2 3
    goodmatchp.clear();
    for (int i = 0; i < matchp23.size(); i++)
    {
        if (matchp23.at(i).size() == 2 && matchp23.at(i).at(0).distance <= 0.6f * matchp23.at(i).at(1).distance)
        {
            goodmatchp.push_back(matchp23[i][0]);
        }
    }

    points1.clear();
    points2.clear();
    for (size_t i = 0; i < goodmatchp.size(); i++)
    {
        points1.push_back(keypoints2[goodmatchp[i].queryIdx].pt);
        points2.push_back(keypoints3[goodmatchp[i].trainIdx].pt);
    }

    Mat H23 = findHomography(points2, points1, RANSAC);

    Mat image23;
    warpPerspective(image3_save, image23, H23, Size(image2.cols * 1.2, image2.rows));
    image2_save.copyTo(image23(Rect(0, 0, image2.cols, image2.rows)));


    orb->detectAndCompute(image23, Mat(), keypoints23, detectp23);

    //1 23
    goodmatchp.clear();
    matcher.knnMatch(detectp1, detectp23, matchp123, 2);

    for (int i = 0; i < matchp123.size(); i++)
    {
        if (matchp123.at(i).size() == 2 && matchp123.at(i).at(0).distance <= 0.6f * matchp123.at(i).at(1).distance)
        {
            goodmatchp.push_back(matchp123[i][0]);
        }
    }


    points1.clear();
    points2.clear();
    for (size_t i = 0; i < goodmatchp.size(); i++)
    {
        points1.push_back(keypoints1[goodmatchp[i].queryIdx].pt);
        points2.push_back(keypoints23[goodmatchp[i].trainIdx].pt);
    }

    Mat H123 = findHomography(points2, points1, RANSAC);

    Mat image123;
    warpPerspective(image3_save, image123, H123, Size(image1.cols * 1.35, image1.rows));
    image1_save.copyTo(image123(Rect(0, 0, image1.cols, image1.rows)));

    orb->detectAndCompute(image123, Mat(), keypoints123, detectp123);
    //123 4

    goodmatchp.clear();
    matcher.knnMatch(detectp123, detectp4, matchp1234, 2);

    for (int i = 0; i < matchp1234.size(); i++)
    {
        if (matchp1234.at(i).size() == 2 && matchp1234.at(i).at(0).distance <= 0.6f * matchp1234.at(i).at(1).distance)
        {
            goodmatchp.push_back(matchp1234[i][0]);
        }
    }

    points1.clear();
    points2.clear();
    for (size_t i = 0; i < goodmatchp.size(); i++)
    {
        points1.push_back(keypoints123[goodmatchp[i].queryIdx].pt);
        points2.push_back(keypoints4[goodmatchp[i].trainIdx].pt);
    }

    Mat H1234 = findHomography(points2, points1, RANSAC);

    Mat image1234;
    warpPerspective(image4_save, image1234, H1234, Size(image123.cols * 1.5, image123.rows));
    image123.copyTo(image1234(Rect(0, 0, image123.cols, image123.rows)));

    imshow("Final Panorama1234", image1234);




    waitKey(0);
    return 0;
}
