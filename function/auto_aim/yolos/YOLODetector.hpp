// YOLODetector.hpp
#pragma once

#include "DetectionResult.hpp"
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

namespace rm_vision {

class YOLODetector {
public:
    // 构造函数：加载模型
    explicit YOLODetector(const std::string& model_path);
    
    // 析构函数
    ~YOLODetector() = default;

    // 核心推理接口
    // frame_id 可选，默认 -1 表示不追踪 ID
    std::vector<DetectionResult> detect(const cv::Mat& frame, int frame_id = -1);

private:
    // OpenVINO 核心组件
    ov::Core core_;
    std::shared_ptr<ov::Model> model_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;

    // 模型参数
    int input_w_ = 480; // 默认值，会在加载时自动更新
    int input_h_ = 480;
    int num_classes_ = 15; // RM 比赛通常 15 类
    
    // 阈值配置
    float conf_threshold_ = 0.50f;
    float nms_threshold_ = 0.45f;

    // 内部辅助函数：Letterbox 预处理
    // 返回处理后的图像以及缩放参数 (scale, dw, dh)
    cv::Mat preprocess(const cv::Mat& src, float& scale, int& w_pad, int& h_pad);
};

} // namespace rm_vision