/**
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#include <dlfcn.h>
#include <unistd.h>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>

#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <NvUffParser.h>

#include "nvdsinfer.h"
#include "nvdsinfer_custom_impl.h"
#include "nvdsinfer_func_utils.h"
#include "nvdsinfer_model_builder.h"
#include "nvdsinfer_utils.h"

namespace nvdsinfer {

/* Default data type for bound layers  - FP32 */
constexpr nvinfer1::DataType kDefaultTensorDataType = nvinfer1::DataType::kFLOAT;

/* Default tensort format for bound layers - Linear. */
constexpr nvinfer1::TensorFormats kDefaultTensorFormats =
    1U << (uint32_t)nvinfer1::TensorFormat::kLINEAR;

CaffeModelParser::CaffeModelParser(const NvDsInferContextInitParams& initParams,
    const std::shared_ptr<DlLibHandle>& handle)
    : BaseModelParser(initParams, handle),
      m_ProtoPath(initParams.protoFilePath),
      m_ModelPath(initParams.modelFilePath)
{
    if(initParams.numOutputLayers <= 0)
    {
        dsInferError("No output layers specified. Need atleast one output layer");
        return;
    }

    for (unsigned int i = 0; i < initParams.numOutputLayers; i++)
    {
        assert(initParams.outputLayerNames[i]);
        m_OutputLayers.emplace_back(initParams.outputLayerNames[i]);
    }
    m_CaffeParser = nvcaffeparser1::createCaffeParser();
}

CaffeModelParser::~CaffeModelParser()
{
    m_CaffeParser.reset();
    /* Destroy the PluginFactory created for building the Caffe model.*/
    if (m_CaffePluginFactory.pluginFactoryV2)
    {
        assert(m_LibHandle);
        auto destroyFunc =
            READ_SYMBOL(m_LibHandle, NvDsInferPluginFactoryCaffeDestroy);
        if (destroyFunc)
        {
            destroyFunc(m_CaffePluginFactory);
        }
        else
        {
            dsInferWarning(
                "Custom lib: %s doesn't have function "
                "<NvDsInferPluginFactoryCaffeDestroy> may cause memory-leak",
                safeStr(m_LibHandle->getPath()));
        }
    }
}

NvDsInferStatus
CaffeModelParser::setPluginFactory()
{
    assert(m_CaffeParser);
    if (!m_LibHandle)
        return NVDSINFER_SUCCESS;

    /* Check if the custom library provides a PluginFactory for Caffe parsing.
     */
    auto fcn = READ_SYMBOL(m_LibHandle, NvDsInferPluginFactoryCaffeGet);
    if (!fcn)
        return NVDSINFER_SUCCESS;

    NvDsInferPluginFactoryType type{PLUGIN_FACTORY_V2};
    if (!fcn(m_CaffePluginFactory, type))
    {
        dsInferError(
            "Could not get PluginFactory instance for "
            "Caffe parsing from custom library");
        return NVDSINFER_CUSTOM_LIB_FAILED;
    }

    if (type != PLUGIN_FACTORY_V2) {
        dsInferError(
                    "Invalid PluginFactory type returned by "
                    "custom library");
            return NVDSINFER_CUSTOM_LIB_FAILED;

    } else {
        m_CaffeParser->setPluginFactoryV2(m_CaffePluginFactory.pluginFactoryV2);
    }

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
CaffeModelParser::parseModel(nvinfer1::INetworkDefinition& network)
{
    if (!isValid())
    {
        dsInferError("parse Caffe model failed, please check config file");
        return NVDSINFER_INVALID_PARAMS;
    }

    if (!file_accessible(m_ProtoPath))
    {
        dsInferError("Cannot access prototxt file '%s'", safeStr(m_ProtoPath));
        return NVDSINFER_CONFIG_FAILED;
    }
    if (!file_accessible(m_ModelPath))
    {
        dsInferError("Cannot access caffemodel file '%s'", safeStr(m_ModelPath));
        return NVDSINFER_CONFIG_FAILED;
    }

    NvDsInferStatus status = setPluginFactory();
    if (status != NVDSINFER_SUCCESS)
    {
        dsInferError("Failed to set caffe plugin Factory from custom lib");
        return NVDSINFER_TENSORRT_ERROR;
    }

    /* Parse the caffe model. */
    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor =
        m_CaffeParser->parse(m_ProtoPath.c_str(), m_ModelPath.c_str(), network,
            nvinfer1::DataType::kFLOAT);

    if (!blobNameToTensor)
    {
        dsInferError("Failed while parsing caffe network: %s", safeStr(m_ProtoPath));
        return NVDSINFER_TENSORRT_ERROR;
    }

    for (const auto& layerName : m_OutputLayers)
    {
        /* Find and mark output layers */
        nvinfer1::ITensor* tensor = blobNameToTensor->find(layerName.c_str());
        if (!tensor)
        {
            dsInferError("Could not find output layer '%s'", safeStr(layerName));
            return NVDSINFER_CONFIG_FAILED;
        }
        network.markOutput(*tensor);
    }

    return NVDSINFER_SUCCESS;
}

UffModelParser::UffModelParser(const NvDsInferContextInitParams& initParams,
    const std::shared_ptr<DlLibHandle>& handle)
    : BaseModelParser(initParams, handle)
{
    m_ModelParams.uffFilePath = initParams.uffFilePath;
    if (string_empty(initParams.uffInputBlobName))
    {
        dsInferError("Uff input blob name is empty");
        return;
    }

    if(initParams.numOutputLayers <= 0)
    {
        dsInferError("No output layers specified. Need atleast one output layer");
        return;
    }

    m_ModelParams.inputNames.emplace_back(initParams.uffInputBlobName);
    nvinfer1::Dims3 uffInputDims(initParams.inferInputDims.c,
        initParams.inferInputDims.h, initParams.inferInputDims.w);
    m_ModelParams.inputDims.emplace_back(uffInputDims);

    if (m_ModelParams.inputDims.size() != m_ModelParams.inputNames.size())
    {
        dsInferError(
            "Unrecognized uff input blob names and dims are not match");
        return;
    }

    switch (initParams.uffInputOrder)
    {
        case NvDsInferTensorOrder_kNCHW:
            m_ModelParams.inputOrder = nvuffparser::UffInputOrder::kNCHW;
            break;
        case NvDsInferTensorOrder_kNHWC:
            m_ModelParams.inputOrder = nvuffparser::UffInputOrder::kNHWC;
            break;
        case NvDsInferTensorOrder_kNC:
            m_ModelParams.inputOrder = nvuffparser::UffInputOrder::kNC;
            break;
        default:
            dsInferError("Unrecognized uff input order");
            m_ModelParams.inputOrder = (nvuffparser::UffInputOrder)(-1);
            return;
    }

    for (unsigned int i = 0; i < initParams.numOutputLayers; i++)
    {
        assert(initParams.outputLayerNames[i]);
        m_ModelParams.outputNames.emplace_back(initParams.outputLayerNames[i]);
    }

    m_UffParser = nvuffparser::createUffParser();
}

UffModelParser::~UffModelParser()
{
    m_UffParser.reset();
}

NvDsInferStatus
UffModelParser::initParser()
{
    /* Register the input layer (name, dims and input order). */
    for (size_t i = 0; i < m_ModelParams.inputNames.size(); ++i)
    {
        if (!m_UffParser->registerInput(m_ModelParams.inputNames[i].c_str(),
                m_ModelParams.inputDims[i], m_ModelParams.inputOrder))
        {
            dsInferError(
                "Failed to register uff input blob: %s DimsCHW:(%s) "
                "Order: %s",
                safeStr(m_ModelParams.inputNames[i]),
                safeStr(dims2Str(m_ModelParams.inputDims[i])),
                (int)m_ModelParams.inputOrder);
            return NVDSINFER_CONFIG_FAILED;
        }
    }

    /* Register outputs. */
    for (const auto& layerName : m_ModelParams.outputNames)
    {
        if (!m_UffParser->registerOutput(layerName.c_str()))
        {
            dsInferError(
                "Failed to register uff output blob: %s", safeStr(layerName));
            return NVDSINFER_CONFIG_FAILED;
        }
    }

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
UffModelParser::parseModel(nvinfer1::INetworkDefinition& network)
{
    if (!isValid())
    {
        dsInferError("parse Uff model failed, please check config file");
        return NVDSINFER_INVALID_PARAMS;
    }

    NvDsInferStatus status = initParser();
    if (status != NVDSINFER_SUCCESS)
    {
        dsInferError("Failed to init uff parser for file: %s",
            safeStr(m_ModelParams.uffFilePath));
        return status;
    }

    if (!file_accessible(m_ModelParams.uffFilePath))
    {
        dsInferError(
            "Cannot access UFF file '%s'", safeStr(m_ModelParams.uffFilePath));
        return NVDSINFER_CONFIG_FAILED;
    }

    if (!m_UffParser->parse(m_ModelParams.uffFilePath.c_str(), network,
            nvinfer1::DataType::kFLOAT))
    {
        dsInferError(
            "Failed to parse UFF file: %s, incorrect file or incorrect"
            " input/output blob names",
            safeStr(m_ModelParams.uffFilePath));
        return NVDSINFER_TENSORRT_ERROR;
    }

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
OnnxModelParser::parseModel(nvinfer1::INetworkDefinition& network)
{
    if (!file_accessible(m_ModelName.c_str()))
    {
        dsInferError("Cannot access ONNX file '%s'", safeStr(m_ModelName));
        return NVDSINFER_CONFIG_FAILED;
    }
    m_OnnxParser = nvonnxparser::createParser(network, *gTrtLogger);

    if (!m_OnnxParser->parseFromFile(
            m_ModelName.c_str(), (int)nvinfer1::ILogger::Severity::kWARNING))
    {
        dsInferError("Failed to parse onnx file");
        return NVDSINFER_TENSORRT_ERROR;
    }
    return NVDSINFER_SUCCESS;
}

CustomModelParser::CustomModelParser(const NvDsInferContextInitParams& initParams,
    const std::shared_ptr<DlLibHandle>& handle)
    : BaseModelParser(initParams, handle)
{
    assert(handle);

    /* Get the address of NvDsInferCreateModelParser interface implemented by
     * the custom library. */
    auto createFcn = READ_SYMBOL(m_LibHandle, NvDsInferCreateModelParser);
    if (!createFcn)
        return;

    /* Create the custom parser using NvDsInferCreateModelParser interface. */
    std::unique_ptr<IModelParser> modelParser(createFcn(&initParams));
    if (!modelParser)
    {
        dsInferError(
            "Failed to create custom parser from lib:%s, model path:%s",
            safeStr(handle->getPath()),
            safeStr(initParams.customNetworkConfigFilePath));
    }

    m_CustomParser = std::move(modelParser);
}

NvDsInferStatus
CustomModelParser::parseModel(nvinfer1::INetworkDefinition& network)
{
    if (!isValid())
    {
        dsInferError(
            "Failed to parse model since parser description is not valid or "
            "parser cannot be created");
        return NVDSINFER_CUSTOM_LIB_FAILED;
    }

    return m_CustomParser->parseModel(network);
}

bool
BuildParams::sanityCheck() const
{
    /* Check for supported network modes. */
    switch (networkMode)
    {
        case NvDsInferNetworkMode_FP32:
        case NvDsInferNetworkMode_FP16:
        case NvDsInferNetworkMode_INT8:
            break;
        default:
            return false;
    }
    return true;
}

bool
ImplicitBuildParams::sanityCheck() const
{
    /* Check for valid batch size. */
    if (maxBatchSize <= 0)
        return false;
    return BuildParams::sanityCheck();
}

NvDsInferStatus
ImplicitBuildParams::configBuilder(TrtModelBuilder& trtBuilder)
{
    return trtBuilder.configImplicitOptions(*this);
}

bool
ExplicitBuildParams::sanityCheck() const
{
    /* Check that min <= opt <= max batch size. */
    if (minBatchSize > optBatchSize || optBatchSize > maxBatchSize)
        return false;

    for (auto& layer : inputProfileDims)
    {
        int nd = -1;
        if (!std::all_of(
                layer.begin(), layer.end(), [&nd](const nvinfer1::Dims& s) {
                    if (nd > 0)
                        return nd == s.nbDims;
                    nd = s.nbDims;
                    return true;
                }))
        {
            dsInferError("Explicit Options sanity check failed.");
            return false;
        }
    }

    return BuildParams::sanityCheck();
}

NvDsInferStatus
ExplicitBuildParams::configBuilder(TrtModelBuilder& trtBuilder)
{
    return trtBuilder.configExplicitOptions(*this);
}

TrtEngine::TrtEngine(UniquePtrWDestroy<nvinfer1::ICudaEngine>&& engine,
    const SharedPtrWDestroy<nvinfer1::IRuntime>& runtime, int dlaCore,
    const std::shared_ptr<DlLibHandle>& dlHandle,
    nvinfer1::IPluginFactory* pluginFactory)
    : m_Runtime(runtime),
      m_Engine(std::move(engine)),
      m_DlHandle(dlHandle),
      m_RuntimePluginFactory(pluginFactory),
      m_DlaCore(dlaCore){}

TrtEngine::~TrtEngine()
{
    m_Engine.reset();

    /* Destroy the Runtime PluginFactory instance if provided. */
    if (m_RuntimePluginFactory && m_DlHandle)
    {
        auto destroyFcn =
            READ_SYMBOL(m_DlHandle, NvDsInferPluginFactoryRuntimeDestroy);
        if (!destroyFcn)
        {
            dsInferWarning(
                "NvDsInferPluginFactoryRuntimeDestroy is missing in custom "
                "lib.");
        }
        destroyFcn(m_RuntimePluginFactory);
    }
    m_Runtime.reset();
}

/* Get properties of bound layers like the name, dimension, datatype
 */
NvDsInferStatus
TrtEngine::getLayerInfo(int idx, NvDsInferLayerInfo& info)
{
    assert(m_Engine);
    assert(idx < m_Engine->getNbBindings());
    nvinfer1::Dims d = m_Engine->getBindingDimensions(idx);

    info.buffer = nullptr;
    info.isInput = m_Engine->bindingIsInput(idx);
    info.bindingIndex = idx;
    info.layerName = safeStr(m_Engine->getBindingName(idx));
    if (m_Engine->hasImplicitBatchDimension())
    {
        info.inferDims = trt2DsDims(d);
    }
    else
    {
        NvDsInferBatchDims batchDims;
        convertFullDims(d, batchDims);
        info.inferDims = batchDims.dims;
    }

    switch (m_Engine->getBindingDataType(idx))
    {
        case nvinfer1::DataType::kFLOAT:
            info.dataType = FLOAT;
            break;
        case nvinfer1::DataType::kHALF:
            info.dataType = HALF;
            break;
        case nvinfer1::DataType::kINT32:
            info.dataType = INT32;
            break;
        case nvinfer1::DataType::kINT8:
            info.dataType = INT8;
            break;
        default:
            dsInferError(
                    "Unknown data type for bound layer i(%s)", safeStr(info.layerName));
            return NVDSINFER_TENSORRT_ERROR;
    }
    return NVDSINFER_SUCCESS;
}

/* Get information for all layers for implicit batch dimensions network. */
NvDsInferStatus
TrtEngine::getImplicitLayersInfo(std::vector<NvDsInferBatchDimsLayerInfo>& layersInfo)
{
    layersInfo.clear();
    int maxBatch = m_Engine->getMaxBatchSize();
    for (int i = 0; i < (int)m_Engine->getNbBindings(); i++)
    {
        NvDsInferBatchDimsLayerInfo layerInfo;
        RETURN_NVINFER_ERROR(getLayerInfo(i, layerInfo),
            "initialize backend context failed on layer: %d", i);
        if (hasWildcard(layerInfo.inferDims))
        {
            dsInferError(
                "ImplicitTrtBackend initialize failed because bindings has "
                "wildcard dims");
            return NVDSINFER_CONFIG_FAILED;
        }
        for (int iSelector = 0; iSelector < (int)kSELECTOR_SIZE; ++iSelector)
        {
            layerInfo.profileDims[iSelector] =
                NvDsInferBatchDims{maxBatch, layerInfo.inferDims};
        }
        layersInfo.emplace_back(layerInfo);
    }
    return NVDSINFER_SUCCESS;
}

/* Get information for all layers for full dimensions network. */
NvDsInferStatus
TrtEngine::getFullDimsLayersInfo(int profileIdx,
        std::vector<NvDsInferBatchDimsLayerInfo>& layersInfo)
{
    layersInfo.clear();
    for (int i = 0; i < (int)m_Engine->getNbBindings(); i++)
    {
        NvDsInferBatchDimsLayerInfo layerInfo;
        RETURN_NVINFER_ERROR(getLayerInfo(i, layerInfo),
            "initialize backend context failed on layer: %d", i);

        if (layerInfo.isInput)
        {
            nvinfer1::Dims minDims = m_Engine->getProfileDimensions(
                    i, profileIdx, nvinfer1::OptProfileSelector::kMIN);
            nvinfer1::Dims optDims = m_Engine->getProfileDimensions(
                    i, profileIdx, nvinfer1::OptProfileSelector::kOPT);
            nvinfer1::Dims maxDims = m_Engine->getProfileDimensions(
                    i, profileIdx, nvinfer1::OptProfileSelector::kMAX);

            assert(minDims <= optDims && optDims <= maxDims);

            NvDsInferBatchDims batchDims;
            convertFullDims(minDims, batchDims);
            layerInfo.profileDims[kSELECTOR_MIN] = batchDims;
            convertFullDims(optDims, batchDims);
            layerInfo.profileDims[kSELECTOR_OPT] = batchDims;
            convertFullDims(maxDims, batchDims);
            layerInfo.profileDims[kSELECTOR_MAX] = batchDims;
        }

        layersInfo.emplace_back(layerInfo);
    }

    return NVDSINFER_SUCCESS;
}

/* Print engine details. */
void
TrtEngine::printEngineInfo()
{
    assert(m_Engine);
    nvinfer1::Dims checkDims = m_Engine->getBindingDimensions(0);
    assert(m_Engine->getNbOptimizationProfiles() > 0);
    std::stringstream s;
    std::vector<NvDsInferBatchDimsLayerInfo> layersInfo;
    bool isFullDims = false;
    if (hasWildcard(checkDims))
    {
        isFullDims = true;
        getFullDimsLayersInfo(0, layersInfo);
        s << "[FullDims Engine Info]: layers num: " << layersInfo.size()
          << "\n";
    }
    else
    {
        isFullDims = false;
        getImplicitLayersInfo(layersInfo);
        s << "[Implicit Engine Info]: layers num: " << layersInfo.size()
          << "\n";
    }

    for (int i = 0; i < (int)layersInfo.size(); ++i)
    {
        NvDsInferBatchDimsLayerInfo& layer = layersInfo[i];
        s << std::setw(3) << std::left << i << " ";
        s << std::setw(6) << std::left << (layer.isInput ? "INPUT" : "OUTPUT")
          << " ";
        s << std::setw(6) << std::left << dataType2Str(layer.dataType) << " ";
        s << std::setw(15) << std::left << safeStr(layer.layerName) << " ";
        s << std::setw(15) << std::left << dims2Str(layer.inferDims) << " ";
        if (isFullDims)
        {
            s << "min: " << std::setw(15) << std::left
              << batchDims2Str(layer.profileDims[kSELECTOR_MIN]) << " ";
            s << "opt: " << std::setw(15) << std::left
              << batchDims2Str(layer.profileDims[kSELECTOR_OPT]) << " ";
            s << "Max: " << std::setw(15) << std::left
              << batchDims2Str(layer.profileDims[kSELECTOR_MAX]) << " ";
        }
        s << "\n";
    }
    dsInferInfo("%s", s.str().c_str());
}

TrtModelBuilder::TrtModelBuilder(int gpuId, nvinfer1::ILogger& logger,
    const std::shared_ptr<DlLibHandle>& dlHandle)
    : m_GpuId(gpuId), m_Logger(logger), m_DlLib(dlHandle)
{
    m_Builder.reset(nvinfer1::createInferBuilder(logger));
    assert(m_Builder);
    m_BuilderConfig.reset(m_Builder->createBuilderConfig());
    assert(m_BuilderConfig);
}

/* Get already built CUDA Engine from custom library. */
std::unique_ptr<TrtEngine>
TrtModelBuilder::getCudaEngineFromCustomLib(NvDsInferCudaEngineGetFcnDeprecated cudaEngineGetDeprecatedFcn,
        NvDsInferEngineCreateCustomFunc cudaEngineGetFcn,
        const NvDsInferContextInitParams& initParams,
        NvDsInferNetworkMode &networkMode)
{
    networkMode = initParams.networkMode;
    nvinfer1::DataType modelDataType;

    switch (initParams.networkMode)
    {
        case NvDsInferNetworkMode_FP32:
        case NvDsInferNetworkMode_FP16:
        case NvDsInferNetworkMode_INT8:
            break;
        default:
            dsInferError("Unknown network mode %d", networkMode);
            return nullptr;
    }

    if (networkMode == NvDsInferNetworkMode_INT8)
    {
        /* Check if platform supports INT8 else use FP16 */
        if (m_Builder->platformHasFastInt8())
        {
            if (m_Int8Calibrator != nullptr)
            {
                /* Set INT8 mode and set the INT8 Calibrator */
                m_BuilderConfig->setFlag(nvinfer1::BuilderFlag::kINT8);
                m_BuilderConfig->setInt8Calibrator(m_Int8Calibrator.get());
                /* modelDataType should be FLOAT for INT8 */
                modelDataType = nvinfer1::DataType::kFLOAT;
            }
            else if (cudaEngineGetFcn != nullptr || cudaEngineGetDeprecatedFcn != nullptr)
            {
                dsInferWarning("INT8 calibration file not specified/accessible. "
                        "INT8 calibration can be done through setDynamicRange "
                        "API in 'NvDsInferCreateNetwork' implementation");
            }
            else
            {
                dsInferWarning("INT8 calibration file not specified. Trying FP16 mode.");
                networkMode = NvDsInferNetworkMode_FP16;
            }
        }
        else
        {
            dsInferWarning("INT8 not supported by platform. Trying FP16 mode.");
            networkMode = NvDsInferNetworkMode_FP16;
        }
    }

    if (networkMode == NvDsInferNetworkMode_FP16)
    {
        /* Check if platform supports FP16 else use FP32 */
        if (m_Builder->platformHasFastFp16())
        {
            m_BuilderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
            modelDataType = nvinfer1::DataType::kHALF;
        }
        else
        {
            dsInferWarning("FP16 not supported by platform. Using FP32 mode.");
            networkMode = NvDsInferNetworkMode_FP32;
        }
    }

    if (networkMode == NvDsInferNetworkMode_FP32)
    {
        modelDataType = nvinfer1::DataType::kFLOAT;
    }

    /* Set the maximum batch size */
    m_Builder->setMaxBatchSize(initParams.maxBatchSize);
    m_BuilderConfig->setMaxWorkspaceSize(kWorkSpaceSize);

    int dla = -1;
    /* Use DLA if specified. */
    if (initParams.useDLA)
    {
        m_BuilderConfig->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        m_BuilderConfig->setDLACore(initParams.dlaCore);
        m_BuilderConfig->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        dla = initParams.dlaCore;

        if (networkMode == NvDsInferNetworkMode_FP32)
        {
            dsInferWarning("FP32 mode requested with DLA. DLA may execute "
                    "in FP16 mode instead.");
        }
    }

    /* Get the  cuda engine from the library */
    nvinfer1::ICudaEngine *engine = nullptr;
    if (cudaEngineGetFcn && (!cudaEngineGetFcn (m_Builder.get(), m_BuilderConfig.get(),
                (NvDsInferContextInitParams *)&initParams,
                modelDataType, engine) ||
            engine == nullptr))
    {
        dsInferError("Failed to create network using custom network creation"
                " function");
        return nullptr;
    }
    if (cudaEngineGetDeprecatedFcn && (!cudaEngineGetDeprecatedFcn (m_Builder.get(),
                (NvDsInferContextInitParams *)&initParams,
                modelDataType, engine) ||
            engine == nullptr))
    {
        dsInferError("Failed to create network using custom network creation"
                " function");
        return nullptr;
    }

    return std::make_unique<TrtEngine>(UniquePtrWDestroy<nvinfer1::ICudaEngine>(engine), dla);
}

/* Build the model and return the generated engine. */
std::unique_ptr<TrtEngine>
TrtModelBuilder::buildModel(const NvDsInferContextInitParams& initParams,
    std::string& suggestedPathName)
{
    std::unique_ptr<TrtEngine> engine;
    std::string modelPath;
    NvDsInferNetworkMode networkMode;

    /* check if custom library provides NvDsInferCudaEngineGet interface. */
    NvDsInferEngineCreateCustomFunc cudaEngineGetFcn = nullptr;
    NvDsInferCudaEngineGetFcnDeprecated cudaEngineGetDeprecatedFcn = nullptr;
    if (m_DlLib && !string_empty(initParams.customEngineCreateFuncName))
    {
        cudaEngineGetFcn = m_DlLib->symbol<NvDsInferEngineCreateCustomFunc>(
                initParams.customEngineCreateFuncName);
        if (!cudaEngineGetFcn)
        {
            dsInferError("Could not find Custom Engine Creation Function '%s' in custom lib",
                    initParams.customEngineCreateFuncName);
            return nullptr;
        }
    }
    if (m_DlLib && cudaEngineGetFcn == nullptr)
        cudaEngineGetDeprecatedFcn = m_DlLib->symbol<NvDsInferCudaEngineGetFcnDeprecated>(
                "NvDsInferCudaEngineGet");

    if (cudaEngineGetFcn || cudaEngineGetDeprecatedFcn ||
            !string_empty(initParams.tltEncodedModelFilePath))
    {
        if (cudaEngineGetFcn || cudaEngineGetDeprecatedFcn)
        {
            /* NvDsInferCudaEngineGet interface provided. */
            char *cwd = getcwd(NULL, 0);
            modelPath = std::string(cwd) + "/model";
            free(cwd);
        }
        else
        {
            /* TLT model. Use NvDsInferCudaEngineGetFromTltModel function
             * provided by nvdsinferutils. */
            cudaEngineGetFcn = NvDsInferCudaEngineGetFromTltModel;
            modelPath = safeStr(initParams.tltEncodedModelFilePath);
        }

        engine = getCudaEngineFromCustomLib (cudaEngineGetDeprecatedFcn,
                cudaEngineGetFcn, initParams, networkMode);
        if (engine == nullptr)
        {
            dsInferError("Failed to get cuda engine from custom library API");
            return nullptr;
        }
    }
    else
    {
        /* Parse the network. */
        NvDsInferStatus status = buildNetwork(initParams);
        if (status != NVDSINFER_SUCCESS)
        {
            dsInferError("failed to build network.");
            return nullptr;
        }

        assert(m_Parser);
        assert(m_Network);
        assert(m_Options);

        /* Build the engine from the parsed network and build parameters. */
        engine = buildEngine();
        if (engine == nullptr)
        {
            dsInferError("failed to build trt engine.");
            return nullptr;
        }
        modelPath = safeStr(m_Parser->getModelName());
        networkMode = m_Options->networkMode;
    }

    std::string devId = std::string("gpu") + std::to_string(m_GpuId);
    if (initParams.useDLA && initParams.dlaCore >= 0)
    {
        devId = std::string("dla") + std::to_string(initParams.dlaCore);
    }

    /* Construct the suggested path for engine file. */
    suggestedPathName =
        modelPath + "_b" + std::to_string(initParams.maxBatchSize) + "_" +
        devId + "_" + networkMode2Str(networkMode) + ".engine";
    return engine;
}

NvDsInferStatus
TrtModelBuilder::buildNetwork(const NvDsInferContextInitParams& initParams)
{
    std::unique_ptr<BaseModelParser> parser;
    assert(m_Builder);

    /* check custom model parser first */
    if (m_DlLib && READ_SYMBOL(m_DlLib, NvDsInferCreateModelParser))
    {
        parser.reset(new CustomModelParser(initParams, m_DlLib));
    }
    /* Check for caffe model files. */
    else if (!string_empty(initParams.modelFilePath) &&
             !string_empty(initParams.protoFilePath))
    {
        parser.reset(new CaffeModelParser(initParams, m_DlLib));
    }
    /* Check for UFF model. */
    else if (!string_empty(initParams.uffFilePath))
    {
        parser.reset(new UffModelParser(initParams, m_DlLib));
    }
    /* Check for Onnx model. */
    else if (!string_empty(initParams.onnxFilePath))
    {
        parser.reset(new OnnxModelParser(initParams, m_DlLib));
    }
    else
    {
        dsInferError(
            "failed to build network since there is no model file matched.");
        return NVDSINFER_CONFIG_FAILED;
    }

    if (!parser || !parser->isValid())
    {
        dsInferError("failed to build network because of invalid parsers.");
        return NVDSINFER_CONFIG_FAILED;
    }

    for(unsigned int i = 0; i < initParams.numOutputIOFormats; ++i)
    {
        assert(initParams.outputIOFormats[i]);
        std::string outputIOFormat(initParams.outputIOFormats[i]);
        size_t pos1 = outputIOFormat.find(":");
        if(pos1 == std::string::npos)
        {
            dsInferError("failed to parse outputIOFormart %s."
            "Expected layerName:type:fmt", initParams.outputIOFormats[i]);
            return NVDSINFER_CONFIG_FAILED;
        }
        size_t pos2 = outputIOFormat.find(":", pos1+1);
        if(pos2 == std::string::npos)
        {
            dsInferError("failed to parse outputIOFormart %s."
            "Expected layerName:type:fmt", initParams.outputIOFormats[i]);
            return NVDSINFER_CONFIG_FAILED;
        }
        std::string layerName = outputIOFormat.substr(0,pos1);
        std::string dataType = outputIOFormat.substr(pos1+1,pos2-pos1-1);
        if(!isValidOutputDataType(dataType))
        {
            dsInferError("Invalid data output datatype specified %s",
            dataType.c_str());
            return NVDSINFER_CONFIG_FAILED;
        }
        std::string format = outputIOFormat.substr(pos2+1);
        if(!isValidOutputFormat(format))
        {
            dsInferError("Invalid output data format specified %s",
            format.c_str());
            return NVDSINFER_CONFIG_FAILED;
        }
    }
    for(unsigned int i = 0; i < initParams.numLayerDevicePrecisions; ++i)
    {
      assert(initParams.layerDevicePrecisions[i]);
      std::string outputDevicePrecision(initParams.layerDevicePrecisions[i]);
      size_t pos1 = outputDevicePrecision.find(":");
      if(pos1 == std::string::npos)
      {
        dsInferError("failed to parse outputDevicePrecision %s."
          "Expected layerName:precisionType:deviceType", initParams.layerDevicePrecisions[i]);
        return NVDSINFER_CONFIG_FAILED;
      }
      size_t pos2 = outputDevicePrecision.find(":", pos1+1);
      if(pos2 == std::string::npos)
      {
        dsInferError("failed to parse outputDevicePrecision %s."
          "Expected layerName:precisionType:deviceType", initParams.layerDevicePrecisions[i]);
        return NVDSINFER_CONFIG_FAILED;
      }
      std::string layerName = outputDevicePrecision.substr(0,pos1);
      std::string precisionType = outputDevicePrecision.substr(pos1+1,pos2-pos1-1);
      if(!isValidPrecisionType(precisionType))
      {
        dsInferError("Invalid output precisionType specified %s",
          precisionType.c_str());
        return NVDSINFER_CONFIG_FAILED;
      }
      std::string deviceType = outputDevicePrecision.substr(pos2+1);
      if(!isValidDeviceType(deviceType))
      {
        dsInferError("Invalid deviceType specified %s",
          deviceType.c_str());
        return NVDSINFER_CONFIG_FAILED;
      }
    }

    std::unique_ptr<BuildParams> buildOptions;
    nvinfer1::NetworkDefinitionCreationFlags netDefFlags = 0;
    /* Create build parameters to build the network as a full dimension network
     * only if the parser supports it and DLA is not to be used. Otherwise build
     * the network as an implicit batch dim network. */
    if (parser->hasFullDimsSupported() &&
            !initParams.forceImplicitBatchDimension)
    {
        netDefFlags |=
            (1U << static_cast<uint32_t>(
                 nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        buildOptions = createDynamicParams(initParams);
    }
    else
    {
        buildOptions = createImplicitParams(initParams);
    }

    UniquePtrWDestroy<nvinfer1::INetworkDefinition> network =
        m_Builder->createNetworkV2(netDefFlags);
    assert(network);

    /* Parse the model using IModelParser interface. */
    NvDsInferStatus status = parser->parseModel(*network);
    if (status != NVDSINFER_SUCCESS)
    {
        dsInferError("failed to build network since parsing model errors.");
        return status;
    }

    assert(!m_Network);
    m_Network = std::move(network);
    m_Options = std::move(buildOptions);
    m_Parser = std::move(parser);
    return NVDSINFER_SUCCESS;
}

/* Create build parameters for implicit batch dim network. */
std::unique_ptr<BuildParams>
TrtModelBuilder::createImplicitParams(const NvDsInferContextInitParams& initParams)
{
    auto params = std::make_unique<ImplicitBuildParams>();

    if (initParams.inferInputDims.c && initParams.inferInputDims.h &&
            initParams.inferInputDims.w)
    {
        params->inputDims.emplace_back(ds2TrtDims(initParams.inferInputDims));
    }

    params->maxBatchSize = initParams.maxBatchSize;
    initCommonParams(*params, initParams);

    return params;
}

/* Create build parameters for full dims network. */
std::unique_ptr<BuildParams>
TrtModelBuilder::createDynamicParams(const NvDsInferContextInitParams& initParams)
{
    auto params = std::make_unique<ExplicitBuildParams>();
    if (initParams.dlaCore < 0 || !initParams.useDLA)
    {
        /* Using GPU */
        params->minBatchSize = 1;
    }
    else
    {
        /* Using DLA */
        params->minBatchSize = initParams.maxBatchSize;
    }

    params->optBatchSize = initParams.maxBatchSize;
    params->maxBatchSize = initParams.maxBatchSize;
    params->inputOrder = initParams.netInputOrder;

    dsInferDebug ("%s: c, h, w = %d, %d, %d, order = %s\n", __func__,
            initParams.inferInputDims.c,
            initParams.inferInputDims.h,
            initParams.inferInputDims.w,
            (params->inputOrder == NvDsInferTensorOrder_kNCHW) ?
            "NCHW" : "NHWC");

    if (initParams.inferInputDims.c && initParams.inferInputDims.h &&
        initParams.inferInputDims.w)
    {
        nvinfer1::Dims dims = ds2TrtDims(initParams.inferInputDims);
        ProfileDims profileDims = {{dims, dims, dims}};
        params->inputProfileDims.emplace_back(profileDims);
    }

    initCommonParams(*params, initParams);
    return params;
}

void
TrtModelBuilder::initCommonParams(BuildParams& params,
        const NvDsInferContextInitParams& initParams)
{
    params.networkMode = initParams.networkMode;
    if (initParams.workspaceSize)
    {
        params.workspaceSize =
            initParams.workspaceSize * UINT64_C(1024) * UINT64_C(1024);
    }
    params.int8CalibrationFilePath = initParams.int8CalibrationFilePath;

    if (initParams.useDLA && initParams.dlaCore >= 0)
        params.dlaCore = initParams.dlaCore;
    else
        params.dlaCore = -1;

    for(unsigned int i=0; i < initParams.numOutputIOFormats; ++i)
    {
        assert(initParams.outputIOFormats[i]);
        std::string outputIOFormat(initParams.outputIOFormats[i]);
        size_t pos1 = outputIOFormat.find(":");
        size_t pos2 = outputIOFormat.find(":", pos1+1);
        std::string layerName = outputIOFormat.substr(0,pos1);
        std::string dataType = outputIOFormat.substr(pos1+1,pos2-pos1-1);
        std::string format = outputIOFormat.substr(pos2+1);
        BuildParams::TensorIOFormat fmt =
        std::make_tuple(str2DataType(dataType),str2TensorFormat(format));
        std::pair<std::string, BuildParams::TensorIOFormat>
            outputFmt{layerName, fmt};
        params.outputFormats.insert(outputFmt);
    }

    for(unsigned int i=0; i < initParams.numLayerDevicePrecisions; ++i)
    {
      assert(initParams.layerDevicePrecisions[i]);
      std::string outputDevicePrecision(initParams.layerDevicePrecisions[i]);
      size_t pos1 = outputDevicePrecision.find(":");
      size_t pos2 = outputDevicePrecision.find(":", pos1+1);
      std::string layerName = outputDevicePrecision.substr(0, pos1);
      std::string precisionType = outputDevicePrecision.substr(pos1+1, pos2-pos1-1);
      std::string deviceType = outputDevicePrecision.substr(pos2+1);
      BuildParams::LayerDevicePrecision fmt =
      std::make_tuple(str2PrecisionType(precisionType),str2DeviceType(deviceType));
      std::pair<std::string, BuildParams::LayerDevicePrecision>
        outputFmt{layerName, fmt};
      params.layerDevicePrecisions.insert(outputFmt);
    }
}

std::unique_ptr<TrtEngine>
TrtModelBuilder::buildEngine()
{
    assert(m_Builder);
    assert(m_Network);
    assert(m_Options);
    assert(m_Parser);
    return buildEngine(*m_Network, *m_Options);
}

std::unique_ptr<TrtEngine>
TrtModelBuilder::buildEngine(nvinfer1::INetworkDefinition& network,
        BuildParams& options)
{
    assert(m_Builder);
    if (!options.sanityCheck())
    {
        dsInferError("build param sanity check failed.");
        return nullptr;
    }

    /* Configure m_BuilderConfig with one of ImplicitBuildParams (configImplicitOptions())
     * or ExplicitBuildParams (configExplicitOptions()).*/
    NvDsInferStatus status = options.configBuilder(*this);
    if (status != NVDSINFER_SUCCESS)
    {
        dsInferError("Failed to configure builder options");
        return nullptr;
    }

    UniquePtrWDestroy<nvinfer1::ICudaEngine> engine =
        m_Builder->buildEngineWithConfig(network, *m_BuilderConfig);

    if (!engine)
    {
        dsInferError("Build engine failed from config file");
        return nullptr;
    }
    return std::make_unique<TrtEngine>(std::move(engine), options.dlaCore);
}

NvDsInferStatus
TrtModelBuilder::configCommonOptions(BuildParams& params)
{
    assert(m_Builder && m_Network && m_BuilderConfig);
    nvinfer1::IBuilder& builder = *m_Builder;
    nvinfer1::INetworkDefinition& network = *m_Network;
    nvinfer1::IBuilderConfig& builderConfig = *m_BuilderConfig;

    int inputLayerNum = network.getNbInputs();
    int outputLayerNum = network.getNbOutputs();

    if(!validateIOTensorNames(params, network))
    {
        dsInferError("Invalid layer name specified for TensorIOFormats");
        return NVDSINFER_CONFIG_FAILED;
    }

    /* Set default datatype and tensor formats for input layers */
    for (int iL = 0; iL < inputLayerNum; iL++)
    {
        nvinfer1::ITensor* input = network.getInput(iL);
        input->setType(kDefaultTensorDataType);
        input->setAllowedFormats(kDefaultTensorFormats);
    }

    /* Set user defined data type and tensor formats for all bound output layers. */
    for (int oL = 0; oL < outputLayerNum; oL++)
    {
        nvinfer1::ITensor* output = network.getOutput(oL);
        if(params.outputFormats.find(output->getName())
            != params.outputFormats.end())
            {
                auto curFmt = params.outputFormats.at(output->getName());
                output->setType(std::get<0>(curFmt));
                output->setAllowedFormats(std::get<1>(curFmt));
            }
    }

    if(!params.layerDevicePrecisions.empty())
    {
      builderConfig.setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);

      for(int idx = 0; idx < network.getNbLayers(); ++idx)
      {
        nvinfer1::ILayer* layer = network.getLayer(idx);

        if(params.layerDevicePrecisions.find(layer->getName())
          != params.layerDevicePrecisions.end())
        {
          auto curType = params.layerDevicePrecisions.at(layer->getName());
          builderConfig.setDeviceType(layer, std::get<1>(curType));
          layer->setPrecision(std::get<0>(curType));
        }
      }
    }

    /* Set workspace size. */
    builderConfig.setMaxWorkspaceSize(params.workspaceSize);

    /* Set the network data type */
    if (params.networkMode == NvDsInferNetworkMode_INT8)
    {
        /* Check if platform supports INT8 else use FP16 */
        if (builder.platformHasFastInt8())
        {
            if (m_Int8Calibrator != nullptr)
            {
                /* Set INT8 mode and set the INT8 Calibrator */
                builderConfig.setFlag(nvinfer1::BuilderFlag::kINT8);
                if (!m_Int8Calibrator)
                {
                    dsInferError("INT8 calibrator not specified.");
                    return NVDSINFER_CONFIG_FAILED;
                }
                builderConfig.setInt8Calibrator(m_Int8Calibrator.get());
            }
            else
            {
                dsInferWarning(
                    "INT8 calibration file not specified. Trying FP16 mode.");
                params.networkMode = NvDsInferNetworkMode_FP16;
            }
        }
        else
        {
            dsInferWarning("INT8 not supported by platform. Trying FP16 mode.");
            params.networkMode = NvDsInferNetworkMode_FP16;
        }
    }

    if (params.networkMode == NvDsInferNetworkMode_FP16)
    {
        /* Check if platform supports FP16 else use FP32 */
        if (builder.platformHasFastFp16())
        {
            builderConfig.setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        else
        {
            dsInferWarning("FP16 not supported by platform. Using FP32 mode.");
            params.networkMode = NvDsInferNetworkMode_FP32;
        }
    }

    /* Set DLA parameters if specified. */
    if (params.dlaCore >= 0)
    {
        if (params.dlaCore >= builder.getNbDLACores())
        {
            dsInferError("DLA core id is not valid, check nvinfer params.");
            return NVDSINFER_CONFIG_FAILED;
        }
        builderConfig.setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        builderConfig.setDLACore(params.dlaCore);
        builderConfig.setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        if (params.networkMode != NvDsInferNetworkMode_INT8)
        {
            // DLA supports only INT8 or FP16
            dsInferWarning("DLA does not support FP32 precision type, using FP16 mode.");
            builderConfig.setFlag(nvinfer1::BuilderFlag::kFP16);
        }
    }

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
TrtModelBuilder::configImplicitOptions(ImplicitBuildParams& params)
{
    assert(m_Builder && m_Network && m_BuilderConfig);
    assert(params.inputDims.size() <= 1);

    nvinfer1::IBuilder& builder = *m_Builder;
    nvinfer1::INetworkDefinition& network = *m_Network;
    nvinfer1::IBuilderConfig& builderConfig = *m_BuilderConfig;

    RETURN_NVINFER_ERROR(configCommonOptions(params),
        "config implicit params failed because of common option's error");

    if (!network.hasImplicitBatchDimension())
    {
        dsInferError(
            "build model failed due to BuildParams(implict) doesn't match "
            "(explicit)network.");
        return NVDSINFER_CONFIG_FAILED;
    }

    if (params.maxBatchSize <= 0)
    {
        dsInferError(
            "build model failed due to maxBatchSize not set for implicit "
            "builder.");
        return NVDSINFER_CONFIG_FAILED;
    }

    builder.setMaxBatchSize(params.maxBatchSize);
    builderConfig.setMaxWorkspaceSize(params.workspaceSize);

    if (!params.inputDims.empty())
    {
        int inputLayerNum = network.getNbInputs();
        for (int iL = 0; iL < inputLayerNum; iL++)
        {
            nvinfer1::ITensor* input = network.getInput(iL);
            // TODO, other input layer dims should not be changed
            // suppose others can be called through initNonImageInputLayers
            if ((int)params.inputDims.size() > iL)
            {
                input->setDimensions(params.inputDims[iL]);
            }
        }
    }
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
TrtModelBuilder::configExplicitOptions(ExplicitBuildParams& params)
{
    assert(m_Builder && m_Network && m_BuilderConfig);
    nvinfer1::IBuilder& builder = *m_Builder;
    nvinfer1::INetworkDefinition& network = *m_Network;
    nvinfer1::IBuilderConfig& builderConfig = *m_BuilderConfig;

    RETURN_NVINFER_ERROR(configCommonOptions(params),
        "config explicit params failed because of common option's error");

    if (network.hasImplicitBatchDimension())
    {
        dsInferError(
            "build model failed due to BuildParams(explicit) doesn't match "
            "(implict)network.");
        return NVDSINFER_CONFIG_FAILED;
    }

    nvinfer1::IOptimizationProfile* profile = builder.createOptimizationProfile();
    assert(profile);
    assert((int)params.inputProfileDims.size() <= network.getNbInputs());

    /* For input layers, set the min/optimal/max dims. */
    int iL = 0;
    for (; iL < (int)params.inputProfileDims.size(); iL++)
    {
        nvinfer1::ITensor* input = network.getInput(iL);
        nvinfer1::Dims modelDims = input->getDimensions(); // include batchSize

        nvinfer1::Dims minDims = params.inputProfileDims.at(
            iL)[(int)nvinfer1::OptProfileSelector::kMIN];

        if (minDims.nbDims + 1 != modelDims.nbDims)
        {
            dsInferError(
                "explict dims.nbDims in config does not match model dims.");
            return NVDSINFER_CONFIG_FAILED;
        }

        if (params.inputOrder == NvDsInferTensorOrder_kNCHW)
        {
            std::move_backward(minDims.d, minDims.d + modelDims.nbDims - 1,
                minDims.d + modelDims.nbDims);
        }
        else if (params.inputOrder == NvDsInferTensorOrder_kNHWC)
        {
            /* For Infer config accept Dims as CHW order by default,
            we need to change it to HWC */
            dsInferDebug ("Switch Dims for NHWC\n");
            minDims.d[3] = minDims.d[0];
        }
        else
        {
            dsInferError ("Unexpected Input Tensor Order\n");
            return NVDSINFER_CONFIG_FAILED;
        }

        minDims.d[0] = params.minBatchSize;
        minDims.nbDims = modelDims.nbDims;
        assert(std::none_of(minDims.d, minDims.d + minDims.nbDims,
            [](int d) { return d < 0; }));
        profile->setDimensions(
            input->getName(), nvinfer1::OptProfileSelector::kMIN, minDims);

        nvinfer1::Dims optDims = params.inputProfileDims.at(
            iL)[(int)nvinfer1::OptProfileSelector::kOPT];
        assert(optDims.nbDims + 1 == modelDims.nbDims);

        if (params.inputOrder == NvDsInferTensorOrder_kNCHW)
        {
            std::move_backward(optDims.d, optDims.d + modelDims.nbDims - 1,
                optDims.d + modelDims.nbDims);
        }
        else
        {   // must be NHWC as already checked above
            optDims.d[3] = optDims.d[0];
        }
        optDims.d[0] = params.optBatchSize;
        optDims.nbDims = modelDims.nbDims;
        assert(std::none_of(optDims.d, optDims.d + optDims.nbDims,
            [](int d) { return d < 0; }));
        profile->setDimensions(
            input->getName(), nvinfer1::OptProfileSelector::kOPT, optDims);

        nvinfer1::Dims maxDims = params.inputProfileDims.at(
            iL)[(int)nvinfer1::OptProfileSelector::kMAX];
        assert(maxDims.nbDims + 1 == modelDims.nbDims);
        if (params.inputOrder == NvDsInferTensorOrder_kNCHW)
            std::move_backward(maxDims.d, maxDims.d + modelDims.nbDims - 1,
                maxDims.d + modelDims.nbDims);
        else
            maxDims.d[3] = maxDims.d[0];
        maxDims.d[0] = params.maxBatchSize;
        maxDims.nbDims = modelDims.nbDims;
        assert(std::none_of(maxDims.d, maxDims.d + maxDims.nbDims,
            [](int d) { return d < 0; }));
        profile->setDimensions(
            input->getName(), nvinfer1::OptProfileSelector::kMAX, maxDims);

        modelDims.d[0] = -1;
        input->setDimensions(modelDims);
    }

    // Todo, just set the other layers same dims as originals
    // Maybe need ask dllib to set other dims and input data
    for (; iL < network.getNbInputs(); ++iL)
    {
        nvinfer1::ITensor* input = network.getInput(iL);
        nvinfer1::Dims modelDims = input->getDimensions(); // include batchSize

        nvinfer1::Dims dims = modelDims;
        dims.d[0] = params.minBatchSize;
        profile->setDimensions(
            input->getName(), nvinfer1::OptProfileSelector::kMIN, dims);

        dims.d[0] = params.optBatchSize;
        profile->setDimensions(
            input->getName(), nvinfer1::OptProfileSelector::kOPT, dims);

        dims.d[0] = params.maxBatchSize;
        profile->setDimensions(
            input->getName(), nvinfer1::OptProfileSelector::kMAX, dims);

        if (std::any_of(
                dims.d, dims.d + dims.nbDims, [](int d) { return d < 0; }))
        {
            dsInferError("Explicit config dims is invalid");
            return NVDSINFER_CONFIG_FAILED;
        }

        dims.d[0] = -1;
        input->setDimensions(dims);
    }

    builderConfig.addOptimizationProfile(profile);

    if (!profile->isValid())
    {
        dsInferError("Explicit Build optimization profile is invalid");
        return NVDSINFER_CONFIG_FAILED;
    }

    return NVDSINFER_SUCCESS;
}

/* Serialize engine and write to file.*/
NvDsInferStatus
TrtModelBuilder::serializeEngine(const std::string& path,
        nvinfer1::ICudaEngine& engine)
{
    std::ofstream fileOut(path, std::ios::binary);
    if (!fileOut.is_open())
    {
        dsInferWarning(
            "Serialize engine failed because of file path: %s opened error",
            safeStr(path));
        return NVDSINFER_TENSORRT_ERROR;
    }

    UniquePtrWDestroy<nvinfer1::IHostMemory> memEngine(engine.serialize());
    if (!memEngine)
    {
        dsInferError("Serialize engine failed to file: %s", safeStr(path));
        return NVDSINFER_TENSORRT_ERROR;
    }

    fileOut.write(static_cast<char*>(memEngine->data()), memEngine->size());
    if (fileOut.fail())
    {
        return NVDSINFER_TENSORRT_ERROR;
    }
    return NVDSINFER_SUCCESS;
}

/* Deserialize engine from file */
std::unique_ptr<TrtEngine>
TrtModelBuilder::deserializeEngine(const std::string& path, int dla)
{
    std::ifstream fileIn(path, std::ios::binary);
    if (!fileIn.is_open())
    {
        dsInferWarning(
            "Deserialize engine failed because file path: %s open error",
            safeStr(path));
        return nullptr;
    }

    fileIn.seekg(0, std::ios::end);
    size_t size = fileIn.tellg();
    fileIn.seekg(0, std::ios::beg);

    std::vector<char> data(size);
    fileIn.read(data.data(), size);
    if (fileIn.fail())
    {
        dsInferError("Deserialize engine failed, file path: %s", safeStr(path));
        return nullptr;
    }

    SharedPtrWDestroy<nvinfer1::IRuntime> runtime(
            nvinfer1::createInferRuntime(m_Logger));
    assert(runtime);

    if (dla > 0)
    {
        runtime->setDLACore(dla);
    }

    nvinfer1::IPluginFactory* factory = nullptr;
    if (m_DlLib)
    {
        auto fcn = READ_SYMBOL(m_DlLib, NvDsInferPluginFactoryRuntimeGet);
        if (fcn && !fcn(factory))
        {
            dsInferError(
                "Deserialize engine failed from file: %s,"
                "because of NvDsInferPluginFactoryRuntimeGet errors",
                safeStr(path));
            return nullptr;
        }
    }

    UniquePtrWDestroy<nvinfer1::ICudaEngine> engine =
        runtime->deserializeCudaEngine(data.data(), size, factory);

    if (!engine)
    {
        dsInferError("Deserialize engine failed from file: %s", safeStr(path));
        return nullptr;
    }
    return std::make_unique<TrtEngine>(
        std::move(engine), runtime, dla, m_DlLib, factory);
}

} // namespace nvdsinfer
