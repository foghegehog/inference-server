#ifndef TENSORRT_BUFFERS_H
#define TENSORRT_BUFFERS_H

#include "NvInfer.h"

#include "common.h"
#include "half.h"

#include "bindingInfo.h"

#include <cassert>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <new>
#include <numeric>
#include <memory>
#include <string>
#include <vector>

namespace inferenceCommon
{

//!
//! \brief  The GenericBuffer class is a templated class for buffers.
//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class handles the allocation,
//!          deallocation, querying of buffers on both the device and the host.
//!          It can handle data of arbitrary types because it stores byte buffers.
//!          The template parameters AllocFunc and FreeFunc are used for the
//!          allocation and deallocation of the buffer.
//!          AllocFunc must be a functor that takes in (void** ptr, size_t size)
//!          and returns bool. ptr is a pointer to where the allocated buffer address should be stored.
//!          size is the amount of memory in bytes to allocate.
//!          The boolean indicates whether or not the memory allocation was successful.
//!          FreeFunc must be a functor that takes in (void* ptr) and returns void.
//!          ptr is the allocated buffer address. It must work with nullptr input.
//!
template <typename AllocFunc, typename FreeFunc>
class GenericBuffer
{
public:
    //!
    //! \brief Construct an empty buffer.
    //!
    GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
        : mSize(0)
        , mCapacity(0)
        , mType(type)
        , mBuffer(nullptr)
    {
    }

    //!
    //! \brief Construct a buffer with the specified allocation size in bytes.
    //!
    GenericBuffer(size_t size, nvinfer1::DataType type)
        : mSize(size)
        , mCapacity(size)
        , mType(type)
    {
        if (!allocFn(&mBuffer, this->nbBytes()))
        {
            throw std::bad_alloc();
        }
    }

    GenericBuffer(GenericBuffer&& buf)
        : mSize(buf.mSize)
        , mCapacity(buf.mCapacity)
        , mType(buf.mType)
        , mBuffer(buf.mBuffer)
    {
        buf.mSize = 0;
        buf.mCapacity = 0;
        buf.mType = nvinfer1::DataType::kFLOAT;
        buf.mBuffer = nullptr;
    }

    GenericBuffer& operator=(GenericBuffer&& buf)
    {
        if (this != &buf)
        {
            freeFn(mBuffer);
            mSize = buf.mSize;
            mCapacity = buf.mCapacity;
            mType = buf.mType;
            mBuffer = buf.mBuffer;
            // Reset buf.
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    void* data()
    {
        return mBuffer;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    const void* data() const
    {
        return mBuffer;
    }

    //!
    //! \brief Returns the size (in number of elements) of the buffer.
    //!
    size_t size() const
    {
        return mSize;
    }

    //!
    //! \brief Returns the size (in bytes) of the buffer.
    //!
    size_t nbBytes() const
    {
        return this->size() * inferenceCommon::getElementSize(mType);
    }

    //!
    //! \brief Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
    //!
    void resize(size_t newSize)
    {
        mSize = newSize;
        if (mCapacity < newSize)
        {
            freeFn(mBuffer);
            if (!allocFn(&mBuffer, this->nbBytes()))
            {
                throw std::bad_alloc{};
            }
            mCapacity = newSize;
        }
    }

    //!
    //! \brief Overload of resize that accepts Dims
    //!
    void resize(const nvinfer1::Dims& dims)
    {
        return this->resize(inferenceCommon::volume(dims));
    }

    ~GenericBuffer()
    {
        freeFn(mBuffer);
    }

private:
    size_t mSize{0}, mCapacity{0};
    nvinfer1::DataType mType;
    void* mBuffer;
    AllocFunc allocFn;
    FreeFunc freeFn;
};

class DeviceAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        return cudaMalloc(ptr, size) == cudaSuccess;
    }
};

class DeviceFree
{
public:
    void operator()(void* ptr) const
    {
        cudaFree(ptr);
    }
};

class HostAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        *ptr = malloc(size);
        return *ptr != nullptr;
    }
};

class HostFree
{
public:
    void operator()(void* ptr) const
    {
        free(ptr);
    }
};

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

//!
//! \brief  The ManagedBuffer class groups together a pair of corresponding device and host buffers.
//!
class ManagedBuffer
{
public:
    DeviceBuffer deviceBuffer;
    HostBuffer hostBuffer;
};

//!
//! \brief  The BufferManager class handles host and device buffer allocation and deallocation.
//!
//! \details This RAII class handles host and device buffer allocation and deallocation,
//!          memcpy between host and device buffers to aid with inference,
//!          and debugging dumps to validate inference. The BufferManager class is meant to be
//!          used to simplify buffer management and any interactions between buffers and the engine.
//!
class BufferManager
{
public:
    static const size_t kINVALID_SIZE_VALUE = ~size_t(0);

    //!
    //! \brief Create a BufferManager for handling buffer interactions.
    //!
    BufferManager(
        const nvinfer1::IExecutionContext* context,
        std::shared_ptr<std::vector<BindingInfo>> bindings,
        const int batchSize = 0)
        : mBindings(bindings)
        , mBatchSize(batchSize)
    {
        // Create host and device buffers
        for (int i = 0; i < mBindings->size(); i++)
        {
            auto dims = context->getBindingDimensions(i);
            size_t vol = !mBatchSize ? 1 : static_cast<size_t>(mBatchSize);
            nvinfer1::DataType type = mBindings->at(i).mBindingDataType;
            int vecDim = mBindings->at(i).mBindingVectorizedDim;
            if (-1 != vecDim) // i.e., 0 != lgScalarsPerVector
            {
                int scalarsPerVec = mBindings->at(i).mBindingComponentsPerElement;
                dims.d[vecDim] = divUp(dims.d[vecDim], scalarsPerVec);
                vol *= scalarsPerVec;
            }
            vol *= inferenceCommon::volume(dims);
            ManagedBuffer* manBuf = new ManagedBuffer();
            manBuf->deviceBuffer = DeviceBuffer(vol, type);
            manBuf->hostBuffer = HostBuffer(vol, type);
            mDeviceBindings.emplace_back(manBuf->deviceBuffer.data());
            mManagedBuffers.emplace_back(std::unique_ptr<ManagedBuffer>(manBuf));
        }
    }

    //!
    //! \brief Returns a vector of device buffers that you can use directly as
    //!        bindings for the execute and enqueue methods of IExecutionContext.
    //!
    std::vector<void*>& getDeviceBindings()
    {
        return mDeviceBindings;
    }

    //!
    //! \brief Returns a vector of device buffers.
    //!
    const std::vector<void*>& getDeviceBindings() const
    {
        return mDeviceBindings;
    }

    //!
    //! \brief Returns the device buffer corresponding to tensorName.
    //!        Returns nullptr if no such tensor can be found.
    //!
    void* getDeviceBuffer(const std::string& tensorName) const
    {
        return getBuffer(false, tensorName);
    }

    //!
    //! \brief Returns the host buffer corresponding to tensorName.
    //!        Returns nullptr if no such tensor can be found.
    //!
    template <typename T>
    T* getHostBuffer(const std::string& tensorName) const
    {
        return static_cast<T*>(getBuffer(true, tensorName));
    }

    int get_binding_index(const std::string& tensorName) const
    {
        int index = -1;
        for (auto i = 0; i < mBindings->size(); i++)
        {
            auto binding = mBindings->at(i);
            if (binding.mBindingName == tensorName)
            {
                index = i;
                break;
            }
        }

        return index;
    }

    //!
    //! \brief Returns the size of the host and device buffers that correspond to tensorName.
    //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
    //!
    size_t size(const std::string& tensorName) const
    {
        auto index = get_binding_index(tensorName);

        if (index == -1)
            return kINVALID_SIZE_VALUE;
        return mManagedBuffers[index]->hostBuffer.nbBytes();
    }

    //!
    //! \brief Dump host buffer with specified tensorName to ostream.
    //!        Prints error message to std::ostream if no such tensor can be found.
    //!
    void dumpBuffer(std::ostream& os, const std::string& tensorName)
    {
        auto index = get_binding_index(tensorName);
        if (index == -1)
        {
            os << "Invalid tensor name" << std::endl;
            return;
        }
        void* buf = mManagedBuffers[index]->hostBuffer.data();
        size_t bufSize = mManagedBuffers[index]->hostBuffer.nbBytes();
        nvinfer1::Dims bufDims = mBindings->at(index).mBindingDimensions;
        size_t rowCount = static_cast<size_t>(bufDims.nbDims > 0 ? bufDims.d[bufDims.nbDims - 1] : mBatchSize);
        int leadDim = mBatchSize;
        int* trailDims = bufDims.d;
        int nbDims = bufDims.nbDims;

        // Fix explicit Dimension networks
        if (!leadDim && nbDims > 0)
        {
            leadDim = bufDims.d[0];
            ++trailDims;
            --nbDims;
        }

        os << "[" << leadDim;
        for (int i = 0; i < nbDims; i++)
            os << ", " << trailDims[i];
        os << "]" << std::endl;
        switch (mBindings->at(index).mBindingDataType)
        {
        case nvinfer1::DataType::kINT32: print<int32_t>(os, buf, bufSize, rowCount); break;
        case nvinfer1::DataType::kFLOAT: print<float>(os, buf, bufSize, rowCount); break;
        case nvinfer1::DataType::kHALF: print<half_float::half>(os, buf, bufSize, rowCount); break;
        case nvinfer1::DataType::kINT8: assert(0 && "Int8 network-level input and output is not supported"); break;
        case nvinfer1::DataType::kBOOL: assert(0 && "Bool network-level input and output are not supported"); break;
        }
    }

    //!
    //! \brief Templated print function that dumps buffers of arbitrary type to std::ostream.
    //!        rowCount parameter controls how many elements are on each line.
    //!        A rowCount of 1 means that there is only 1 element on each line.
    //!
    template <typename T>
    void print(std::ostream& os, void* buf, size_t bufSize, size_t rowCount)
    {
        assert(rowCount != 0);
        assert(bufSize % sizeof(T) == 0);
        T* typedBuf = static_cast<T*>(buf);
        size_t numItems = bufSize / sizeof(T);
        for (int i = 0; i < static_cast<int>(numItems); i++)
        {
            // Handle rowCount == 1 case
            if (rowCount == 1 && i != static_cast<int>(numItems) - 1)
                os << typedBuf[i] << std::endl;
            else if (rowCount == 1)
                os << typedBuf[i];
            // Handle rowCount > 1 case
            else if (i % rowCount == 0)
                os << typedBuf[i];
            else if (i % rowCount == rowCount - 1)
                os << " " << typedBuf[i] << std::endl;
            else
                os << " " << typedBuf[i];
        }
    }

    //!
    //! \brief Copy the contents of input host buffers to input device buffers synchronously.
    //!
    void copyInputToDevice()
    {
        memcpyBuffers(true, false, false);
    }

    //!
    //! \brief Copy the contents of output device buffers to output host buffers synchronously.
    //!
    void copyOutputToHost()
    {
        memcpyBuffers(false, true, false);
    }

    //!
    //! \brief Copy the contents of input host buffers to input device buffers asynchronously.
    //!
    void copyInputToDeviceAsync(const cudaStream_t& stream = 0)
    {
        memcpyBuffers(true, false, true, stream);
    }

    //!
    //! \brief Copy the contents of output device buffers to output host buffers asynchronously.
    //!
    void copyOutputToHostAsync(const cudaStream_t& stream = 0)
    {
        memcpyBuffers(false, true, true, stream);
    }

    ~BufferManager() = default;

private:
    void* getBuffer(const bool isHost, const std::string& tensorName) const
    {
        auto index = get_binding_index(tensorName);
        if (index == -1)
            return nullptr;
        return (isHost ? mManagedBuffers[index]->hostBuffer.data() : mManagedBuffers[index]->deviceBuffer.data());
    }

    void memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream = 0)
    {
        for (int i = 0; i < mBindings->size(); i++)
        {
            void* dstPtr
                = deviceToHost ? mManagedBuffers[i]->hostBuffer.data() : mManagedBuffers[i]->deviceBuffer.data();
            const void* srcPtr
                = deviceToHost ? mManagedBuffers[i]->deviceBuffer.data() : mManagedBuffers[i]->hostBuffer.data();
            const size_t byteSize = mManagedBuffers[i]->hostBuffer.nbBytes();
            const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
            if ((copyInput && mBindings->at(i).mIsInput) || (!copyInput && !mBindings->at(i).mIsInput))
            {
                if (async)
                    CHECK(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream));
                else
                    CHECK(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType));
            }
        }
    }

    std::shared_ptr<std::vector<BindingInfo>> mBindings;
    int mBatchSize;                                              //!< The batch size for legacy networks, 0 otherwise.
    std::vector<std::unique_ptr<ManagedBuffer>> mManagedBuffers; //!< The vector of pointers to managed buffers
    std::vector<void*> mDeviceBindings; //!< The vector of device buffers needed for engine execution
};

} // namespace inferenceCommon

#endif // TENSORRT_BUFFERS_H