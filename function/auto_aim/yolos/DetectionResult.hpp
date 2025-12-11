// DetectionResult.hpp
#pragma once

#include <opencv2/core/types.hpp>
#include <vector>

namespace rm_vision {

struct DetectionResult {
    int cls;             // 类别 ID (0-14)
    float score;         // 置信度
    cv::Rect bbox;       // 目标框 (x, y, w, h) - 基于原图尺寸
    cv::Point2f center;  // 目标中心点 (x, y) - 基于原图尺寸
    
    // 可选：如果你的模型输出了关键点 (Pose)，存在这里
    std::vector<cv::Point2f> kpts; 
    
    // 帧 ID，用于多线程或卡尔曼滤波的时间对齐
    int frame_id;
};

} // namespace rm_vision