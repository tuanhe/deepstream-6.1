/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "nvdsinfer_custom_impl.h"
#include "trt_utils.h"

static const int NUM_CLASSES_YOLO = 80;

extern "C" bool NvDsInferParseCustomYoloV5(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

static void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
                                 float& bestConf, int& bestClassId)
{
    // first 5 element are box and obj confidence
    bestClassId = 5;
    bestConf = 0;

    for (int i = 5; i < numClasses + 5; i++)
    {
        if (it[i] > bestConf)
        {
            bestConf = it[i];
            bestClassId = i - 5;
        }
    }

}

static bool NvDsInferParseYoloV5(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList,
    const std::vector<float> &anchors,
    const std::vector<std::vector<int>> &masks)
{
    //std::cout << "Entering the " << __FUNCTION__ << "()\n";
    //std::cout << "output layer size : " << outputLayersInfo.size() << "\n";
    //std::cout << "output layer name : " << outputLayersInfo[0].layerName << "\n";
    //std::cout << "output layer element number : " << outputLayersInfo[0].inferDims.numElements << "\n";
    //std::cout << "detectionParams  : " << detectionParams.perClassThreshold[0] << "\n";
    //std::cout << "output layer dims: " << outputLayersInfo[0].inferDims.numDims << ":\n       [";
    //for(uint32_t i =0; i < outputLayersInfo[0].inferDims.numDims; ++i)
    //    std::cout << " " << outputLayersInfo[0].inferDims.d[i];
    //std::cout << " ]" <<std::endl;

    const float confThreshold = 0.25;

    int64_t count = 1;
    for(uint32_t i =0; i < outputLayersInfo[0].inferDims.numDims; ++i)
        count *= outputLayersInfo[0].inferDims.d[i];
    
    std::vector<float> output((float*)outputLayersInfo[0].buffer, (float*)outputLayersInfo[0].buffer + count);
    std::vector<int> outputShape{1,25200, 85};

    int numClasses = outputLayersInfo[0].inferDims.d[1] - 5;
    int elementsInBatch = outputLayersInfo[0].inferDims.numElements;

    for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape[2])
    {
        float boxConf = it[4];
        //std::cout << ++idx << " " << boxConf << std::endl; 
        if (boxConf <= confThreshold)
            continue;

        float objConf;
        int classId;
        getBestClassInfo(it, numClasses, objConf, classId);
        
        float confidence = boxConf * objConf;
        if(confidence < confThreshold)
            continue;

        int centerX = (int) (it[0]);
        int centerY = (int) (it[1]);
        int width = (int) (it[2]);
        int height = (int) (it[3]);
        int left = centerX - width / 2;
        int top = centerY - height / 2;

        NvDsInferParseObjectInfo object;
        object.width = width;
        object.height = height;
        object.left = left;
        object.top = top;
        object.detectionConfidence = confidence;
        object.classId = classId;
        objectList.emplace_back(object);
    }
    return true;
}

/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseCustomYoloV5(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    static const std::vector<float> kANCHORS = {
        10.0, 13.0, 16.0,  30.0,  33.0, 23.0,  30.0,  61.0,  62.0,
        45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};
    static const std::vector<std::vector<int>> kMASKS = {
        {6, 7, 8},
        {3, 4, 5},
        {0, 1, 2}};
    return NvDsInferParseYoloV5 (
        outputLayersInfo, networkInfo, detectionParams, objectList,
        kANCHORS, kMASKS);
}


/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV5);
