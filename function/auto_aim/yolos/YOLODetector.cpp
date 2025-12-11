
#include "YOLODetector.hpp"
#include <iostream>
#include <algorithm>

namespace rm_vision {

YOLODetector::YOLODetector(const std::string& model_path) {
    // 1. 初始化 Core
    // ov::Core 自动管理插件
    
    // 2. 读取模型
    model_ = core_.read_model(model_path);

    // 3. 获取输入尺寸信息 (通常 YOLOv8 是动态或固定的 [1,3,640,640])
    // 我们强制获取 Width 和 Height
    const auto& input_shape = model_->input().get_shape();
    input_h_ = input_shape[2];
    input_w_ = input_shape[3];
    
    // 4. 获取输出信息以确认类别数
    // YOLOv8 输出通常是 [1, 4+Classes, 8400]
    const auto& output_shape = model_->output().get_shape();
    int channels = output_shape[1]; 
    num_classes_ = channels - 4; // cx,cy,w,h + classes

    // 5. 编译模型 (Latency 模式适合 RM 自瞄)
    compiled_model_ = core_.compile_model(model_, "CPU", 
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    
    // 6. 创建推理请求 (复用此对象以提升速度)
    infer_request_ = compiled_model_.create_infer_request();
}

cv::Mat YOLODetector::preprocess(const cv::Mat& src, float& scale, int& w_pad, int& h_pad) {
    int w = src.cols;
    int h = src.rows;

    // 计算缩放比例 (取最小比例，保证图片能完整塞进去)
    scale = std::min((float)input_w_ / w, (float)input_h_ / h);

    int new_w = w * scale;
    int new_h = h * scale;

    // 计算 Padding (居中放置)
    w_pad = (input_w_ - new_w) / 2;
    h_pad = (input_h_ - new_h) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));

    // 填充灰色边框 (114 是 YOLO 训练时的标准底色)
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, 
                       h_pad, input_h_ - new_h - h_pad, 
                       w_pad, input_w_ - new_w - w_pad, 
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    
    return padded;
}

std::vector<DetectionResult> YOLODetector::detect(const cv::Mat& frame, int frame_id) {
    if (frame.empty()) return {};

    // ---------------- STEP 1: Preprocess ----------------
    float scale;
    int w_pad, h_pad;
    cv::Mat input_img = preprocess(frame, scale, w_pad, h_pad);

    // Blob 转换: [H, W, C] -> [N, C, H, W], BGR -> RGB, /255.0
    // OpenCV 的 blobFromImage 可以高效完成这些
    cv::Mat blob;
    cv::dnn::blobFromImage(input_img, blob, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false);

    // ---------------- STEP 2: Inference ----------------
    // 获取输入 tensor 指针并填充数据
    ov::Tensor input_tensor = infer_request_.get_input_tensor();
    // blob.data 是连续的 float 内存，直接拷贝
    std::memcpy(input_tensor.data(), blob.data, input_w_ * input_h_ * 3 * sizeof(float));

    // 执行推理
    infer_request_.infer();

    // ---------------- STEP 3: Post-process (Decode) ----------------
    const ov::Tensor& output_tensor = infer_request_.get_output_tensor();
    const float* raw_output = output_tensor.data<const float>();
    const auto& out_shape = output_tensor.get_shape(); 
    // out_shape: [1, 4+nc, 8400]

    int dims = out_shape[1];      // 4 + classes
    int anchors = out_shape[2];   // 8400

    // YOLOv8 输出是 [1, 19, 8400]，为了方便处理，我们需要转置成 [8400, 19]
    // 使用 OpenCV Mat 进行转置 (View -> Transpose)
    cv::Mat output_buffer(dims, anchors, CV_32F, (void*)raw_output);
    cv::Mat output_t = output_buffer.t(); // Transpose: [8400, dims]
    const float* data = (float*)output_t.data;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // 遍历所有 Anchors
    for (int i = 0; i < anchors; ++i) {
        const float* row = data + i * dims;
        
        // 查找最大置信度的类别
        // row[0-3] 是 bbox, row[4...] 是 scores
        const float* scores = row + 4;
        
        // 快速找最大值及其索引
        // std::max_element 稍微慢一点，手写循环更快
        float max_score = -1.0f;
        int max_class_id = -1;
        for (int c = 0; c < num_classes_; ++c) {
            if (scores[c] > max_score) {
                max_score = scores[c];
                max_class_id = c;
            }
        }

        if (max_score > conf_threshold_) {
            // 解析坐标 (cx, cy, w, h) - 此时是基于 640x640 的
            float cx = row[0];
            float cy = row[1];
            float w = row[2];
            float h = row[3];

            // 映射回原图坐标 (关键步骤)
            // (x_net - pad) / scale
            int left = static_cast<int>((cx - 0.5 * w - w_pad) / scale);
            int top = static_cast<int>((cy - 0.5 * h - h_pad) / scale);
            int width = static_cast<int>(w / scale);
            int height = static_cast<int>(h / scale);

            // 边界保护
            left = std::max(0, left);
            top = std::max(0, top);
            width = std::min(width, frame.cols - left);
            height = std::min(height, frame.rows - top);

            boxes.emplace_back(left, top, width, height);
            confidences.push_back(max_score);
            class_ids.push_back(max_class_id);
        }
    }

    // ---------------- STEP 4: NMS ----------------
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);

    // ---------------- STEP 5: Pack Results ----------------
    std::vector<DetectionResult> results;
    for (int idx : indices) {
        DetectionResult res;
        res.cls = class_ids[idx];
        res.score = confidences[idx];
        res.bbox = boxes[idx];
        res.center = cv::Point2f(res.bbox.x + res.bbox.width / 2.0f, 
                                 res.bbox.y + res.bbox.height / 2.0f);
        res.frame_id = frame_id;
        results.push_back(res);
    }

    return results;
}

} // namespace rm_vision