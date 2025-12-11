#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include "MvCameraControl.h"

class HikDriver {
public:
    void* handle = nullptr;
    unsigned char* pData = nullptr;
    unsigned int nDataSize = 0;
    bool is_open = false;

    HikDriver() {}
    
    ~HikDriver() {
        if (handle) {
            MV_CC_StopGrabbing(handle);
            MV_CC_CloseDevice(handle);
            MV_CC_DestroyHandle(handle);
        }
        if (pData) free(pData);
    }

    bool init() {
        int nRet = MV_OK;
        MV_CC_DEVICE_INFO_LIST stDeviceList;
        memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

        // 枚举设备
        nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
        if (MV_OK != nRet || stDeviceList.nDeviceNum == 0) {
            std::cerr << "[HikDriver] No camera found!" << std::endl;
            return false;
        }

        // 创建句柄 & 打开设备
        nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[0]);
        if (MV_OK != nRet) return false;
        nRet = MV_CC_OpenDevice(handle);
        if (MV_OK != nRet) return false;

        // --- 核心配置：关闭自动，手动控制 ---
        MV_CC_SetEnumValue(handle, "TriggerMode", 0);       // 连续采集
        MV_CC_SetEnumValue(handle, "ExposureAuto", 0);      // 关闭自动曝光
        MV_CC_SetEnumValue(handle, "GainAuto", 0);          // 关闭自动增益
        MV_CC_SetEnumValue(handle, "BalanceWhiteAuto", 0);  // 关闭自动白平衡

        // 默认参数 (根据你的现场环境微调)
        setExposureTime(3000.0f); 
        setGain(12.0f);

        // 开始采集
        nRet = MV_CC_StartGrabbing(handle);
        if (MV_OK != nRet) return false;

        // 预分配内存 (20MB 足够 1280x1024 RGB)
        nDataSize = 20 * 1024 * 1024; 
        pData = (unsigned char*)malloc(nDataSize);
        is_open = true;
        
        std::cout << "[HikDriver] Camera Initialized." << std::endl;
        return true;
    }

    void setExposureTime(float val) {
        if (!is_open) return;
        if (val < 100) val = 100;
        MV_CC_SetFloatValue(handle, "ExposureTime", val);
    }

    void setGain(float val) {
        if (!is_open) return;
        if (val < 0) val = 0; if (val > 20) val = 20;
        MV_CC_SetFloatValue(handle, "Gain", val);
    }

    bool read(cv::Mat& frame) {
        if (!is_open) return false;
        
        MV_FRAME_OUT_INFO_EX stImageInfo = {0};
        // 超时 1000ms
        int nRet = MV_CC_GetOneFrameTimeout(handle, pData, nDataSize, &stImageInfo, 1000);
        
        if (nRet == MV_OK) {
            // 像素格式转换 (Bayer/YUV -> BGR)
            MV_CC_PIXEL_CONVERT_PARAM stConvertParam = {0};
            stConvertParam.nWidth = stImageInfo.nWidth;
            stConvertParam.nHeight = stImageInfo.nHeight;
            stConvertParam.pSrcData = pData;
            stConvertParam.nSrcDataLen = stImageInfo.nFrameLen;
            stConvertParam.enSrcPixelType = stImageInfo.enPixelType;
            stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
            
            if (frame.empty() || frame.cols != stImageInfo.nWidth || frame.rows != stImageInfo.nHeight) {
                frame = cv::Mat(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC3);
            }
            stConvertParam.pDstBuffer = frame.data;
            stConvertParam.nDstBufferSize = frame.total() * frame.elemSize();
            
            MV_CC_ConvertPixelType(handle, &stConvertParam);
            return true;
        }
        return false;
    }
};