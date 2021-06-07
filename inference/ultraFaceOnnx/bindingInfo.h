#ifndef BINDING_INFO_H
#define BINDING_INFO_H

#include "NvInfer.h"

#include <string>

struct BindingInfo
{
    BindingInfo(
        nvinfer1::DataType dataType,
        nvinfer1::Dims bindingDimensions,
        int32_t vectorizedDim,
        int32_t componentsPerElement,
        const char * bindingName,
        bool isInput)
        :mBindingDataType(dataType),
        mBindingDimensions(bindingDimensions),
        mBindingVectorizedDim(vectorizedDim), 
        mBindingComponentsPerElement(componentsPerElement),
        mBindingName(bindingName),
        mIsInput(isInput)
    {
    }

    nvinfer1::DataType mBindingDataType;
    nvinfer1::Dims mBindingDimensions;
    int32_t mBindingVectorizedDim;
    int32_t mBindingComponentsPerElement;
    std::string mBindingName;
    bool mIsInput;
};

#endif