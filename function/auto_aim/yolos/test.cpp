#include <iostream>
#include <algorithm>
#include <chrono>

#include "HikDriver.hpp"
#include "YOLODetector.hpp"

using namespace rm_vision;

// ================= 赛场参数配置 =================
const std::string XML_PATH = "../models/best.xml"; // 注意路径
const float SEARCH_CONF = 0.40f; // 搜寻阈值 (低)
const float TRACK_CONF  = 0.60f; // 追踪阈值 (高)
const float ROI_SCALE   = 2.0f;  // ROI 放大倍率
const int MAX_LOST_CNT  = 10;    // 最大丢失帧数

// 辅助：限制 Rect 不越界
cv::Rect make_safe_rect(const cv::Rect& r, int max_w, int max_h) {
    int x = std::max(0, r.x);
    int y = std::max(0, r.y);
    int w = std::min(r.width, max_w - x);
    int h = std::min(r.height, max_h - y);
    return cv::Rect(x, y, w, h);
}

int main() {
    // 1. 初始化模块
    HikDriver camera;
    if (!camera.init()) return -1;

    YOLODetector detector(XML_PATH);

    // 2. 状态机变量
    bool is_tracking = false;
    cv::Rect last_rect;          // 上一帧目标在全图的位置
    cv::Point2f velocity(0, 0);  // 速度预测 (dx, dy)
    int lost_count = 0;          // 丢失计数
    int target_id = -1;          // 锁定的目标 ID

    cv::Mat frame;
    int frame_id = 0;

    // FPS 计算
    auto fps_start = std::chrono::steady_clock::now();
    int fps_frame_cnt = 0;
    float current_fps = 0.0f;

    std::cout << "=== RM Vision System Started ===" << std::endl;
    std::cout << "Keys: [U] Exp Up, [J] Exp Down, [ESC] Quit" << std::endl;

    while (true) {
        // --- 硬件采集 ---
        if (!camera.read(frame)) continue;
        
        frame_id++;
        fps_frame_cnt++;
        
        // FPS 更新
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - fps_start).count() >= 1000) {
            current_fps = fps_frame_cnt;
            fps_frame_cnt = 0;
            fps_start = now;
            std::cout << "FPS: " << current_fps << std::endl;
        }

        // --- 核心 ROI 逻辑 ---
        std::vector<DetectionResult> results;
        cv::Rect roi_rect; // 如果处于追踪模式，记录 ROI 在原图的偏移

        if (is_tracking) {
            // [TRACK 模式]
            // 1. 预测位置：中心点 + 速度
            cv::Point center = (last_rect.tl() + last_rect.br()) / 2;
            center.x += (int)velocity.x;
            center.y += (int)velocity.y;

            // 2. 生成 ROI 框 (根据上次目标大小放大)
            int side = std::max(last_rect.width, last_rect.height) * ROI_SCALE;
            if (side < 128) side = 128; // 最小尺寸保护

            roi_rect = cv::Rect(center.x - side/2, center.y - side/2, side, side);
            roi_rect = make_safe_rect(roi_rect, frame.cols, frame.rows);

            // 3. 抠图 & 推理
            // clone() 是必须的，保证内存连续性
            cv::Mat roi_img = frame(roi_rect).clone(); 
            
            // 注意：这里我们不需要手动改 detector 的阈值，
            // 而是拿到结果后自己筛选。Detector 默认 0.5，我们可以接受。
            auto raw_results = detector.detect(roi_img, frame_id);

            // 4. 结果筛选 & 坐标还原
            for (auto& res : raw_results) {
                // 还原坐标：ROI 坐标 -> 全图坐标
                res.bbox.x += roi_rect.x;
                res.bbox.y += roi_rect.y;
                res.center.x += roi_rect.x;
                res.center.y += roi_rect.y;

                // 筛选：ID 必须匹配，且置信度要高 (Track 模式要求严)
                if (res.cls == target_id && res.score > TRACK_CONF) {
                    results.push_back(res);
                }
            }

            // 状态更新
            if (results.empty()) {
                lost_count++;
                if (lost_count > MAX_LOST_CNT) {
                    is_tracking = false; // 彻底丢失，切回全图
                    std::cout << "[WARN] Target Lost. Switching to Search Mode." << std::endl;
                }
            } else {
                lost_count = 0;
            }

        } else {
            // [SEARCH 模式]
            auto raw_results = detector.detect(frame, frame_id);
            
            // 筛选：只要置信度 > SEARCH_CONF 即可 (Search 模式要求宽)
            for (auto& res : raw_results) {
                if (res.score > SEARCH_CONF) {
                    results.push_back(res);
                }
            }
        }

        // --- 决策层 (选最佳目标) ---
        if (!results.empty()) {
            // 按置信度排序 (也可以按距离中心最近排序)
            std::sort(results.begin(), results.end(), [](const DetectionResult& a, const DetectionResult& b) {
                return a.score > b.score;
            });

            const auto& best = results[0];

            // 更新追踪状态
            if (is_tracking) {
                // 更新速度 (简单平滑)
                cv::Point old_center = (last_rect.tl() + last_rect.br()) / 2;
                cv::Point new_center = best.center;
                velocity = cv::Point2f(new_center - old_center);
            } else {
                velocity = cv::Point2f(0, 0); // 刚发现目标，速度未知
            }

            is_tracking = true;
            last_rect = best.bbox;
            target_id = best.cls;
        }

        // --- 可视化 (仅用于调试) ---
        // 绘制 ROI 框 (黄色)
        if (is_tracking) {
            cv::rectangle(frame, roi_rect, cv::Scalar(0, 255, 255), 2);
            cv::putText(frame, "ROI TRACK", roi_rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
        }

        // 绘制目标框 (红/蓝)
        for (const auto& res : results) {
            cv::Scalar color = (res.cls >= 7 && res.cls <= 14) ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
            cv::rectangle(frame, res.bbox, color, 2);
            std::string label = std::to_string(res.cls) + " " + std::to_string((int)(res.score * 100)) + "%";
            cv::putText(frame, label, res.bbox.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }

        // 绘制 FPS
        cv::putText(frame, "FPS: " + std::to_string((int)current_fps), {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        cv::imshow("RM War Vision", frame);

        // --- 键盘控制 (调节曝光) ---
        int key = cv::waitKey(1);
        if (key == 27) break; // ESC
        if (key == 'u' || key == 'U') camera.setExposureTime(3000 + 500); // 示例：简单增加
        if (key == 'j' || key == 'J') camera.setExposureTime(3000 - 500);
    }

    return 0;
}