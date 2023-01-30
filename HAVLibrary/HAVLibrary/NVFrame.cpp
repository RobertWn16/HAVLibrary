#include "NVFrame.hpp"
#include "ColorSpaceConversion.cuh"
#include <fstream>
constexpr unsigned int RGB_NO_OF_CHANNELS = 3;

static winrt::hresult CUDAHr(cudaError_t cudaErr)
{
    if (cudaErr < cudaSuccess)
        return E_FAIL;

    return S_OK;
}

static winrt::hresult CUHr(CUresult cuRes)
{
    if (cuRes != CUDA_SUCCESS)
        return E_FAIL;

    return S_OK;
}

static double GetChannelFactor(HVFormat format)
{
    switch (format)
    {
    case HV_FORMAT_BGRA64_HDR10:
        return 8.0f;
    case HV_FORMAT_RGBA8:
    case HV_FORMAT_BGRA32:
        return 4.0f;
        break;
    case HV_FORMAT_RGB8:
        return 3.0f;
        break;
    case HV_FORMAT_NV12:
        return 1.5f;
        break;
    case HV_FORMAT_P016:
        return 3.0f;
        break;
    default:
        break;
    }
}

void NVFrame::NV12_BGRX8(NVFrame* out, bool inverted, bool exAlpha, int value)
{
    unsigned char* npImage[2];
    npImage[0] = reinterpret_cast<unsigned char*>(cuFrame);
    npImage[1] = reinterpret_cast<unsigned char*>(cuFrame) + cuDesc.width * cuDesc.height;

    hav_nv12_bgra32_SDR(npImage[0], npImage[1], cuDesc.width, cuDesc.height, false, 0.247138f, 0.0555243f, reinterpret_cast<unsigned char*>(out->cuFrame), true, 255.0f);
}

void NVFrame::P016_BGRX8(NVFrame* out, bool inverted, bool exAlpha, int value)
{
    unsigned short* npImage[2];
    npImage[0] = reinterpret_cast<unsigned short*>(cuFrame);
    npImage[1] = reinterpret_cast<unsigned short*>(cuFrame) + cuDesc.width * cuDesc.height;

    hav_p016_HDR10_bgra32_SDR(npImage[0], npImage[1], cuDesc.width, cuDesc.height, reinterpret_cast<unsigned char*>(out->cuFrame), value, false);
}

void NVFrame::PO16_BGRX16(NVFrame* out, unsigned int bitdepth, bool inverted, bool exAlpha, int alphaValue)
{
    unsigned short* npImage[2];
    npImage[0] = reinterpret_cast<unsigned short*>(cuFrame);
    npImage[1] = reinterpret_cast<unsigned short*>(cuFrame) + cuDesc.width * cuDesc.height;

    hav_p016_HDR10_bgra64_HDR10_PQ_ACES(npImage[0], npImage[1], cuDesc.width, cuDesc.height, 0.0f, 2000.0f, 344.0f, 0.247138f, 0.0555243f,  reinterpret_cast<unsigned short*>(out->cuFrame), true, alphaValue);
    //hav_p016_HDR10_bgra64_HDR10_Linear(npImage[0], npImage[1], cuDesc.width, cuDesc.height, false, 344.0f, 0.247138f, 0.0555243f ,reinterpret_cast<unsigned short*>(out->cuFrame), true, alphaValue);
}

winrt::hresult NVFrame::GetDesc(FRAME_OUTPUT_DESC& desc)
{
    desc = cuDesc;
    return S_OK;
}

winrt::hresult NVFrame::ConvertFormat(HVFormat fmt, IFrame *out)
{
    FRAME_OUTPUT_DESC out_frame_desc = { };
    winrt::check_pointer(out);
    out->GetDesc(out_frame_desc);
    
    if (out_frame_desc.width != cuDesc.width || out_frame_desc.height != cuDesc.height)
        return E_INVALIDARG;
    if (out_frame_desc.format == cuDesc.format)
        return E_INVALIDARG;

    if (cuDesc.format == HV_FORMAT_NV12) {
        switch (out_frame_desc.format)
        {
        case HV_FORMAT_RGBA8:
            NV12_BGRX8(dynamic_cast<NVFrame*>(out), true, true);
            break;
        case HV_FORMAT_RGB8:
            NV12_BGRX8(dynamic_cast<NVFrame*>(out), true);
            break;
        case HV_FORMAT_BGRA32:
            NV12_BGRX8(dynamic_cast<NVFrame*>(out), false, true, 255);
            break;
        case HV_FORMAT_NV12:
            break;
        case HV_FORMAT_P016:
            break;
        default:
            break;
        }
        return S_OK;
    }

    if (cuDesc.format == HV_FORMAT_P016)
    {
        switch (out_frame_desc.format)
        {
        case HV_FORMAT_BGRA32:
            P016_BGRX8(dynamic_cast<NVFrame*>(out), false, true, 255);
            break;
        case HV_FORMAT_BGRA64_HDR10:
            PO16_BGRX16(dynamic_cast<NVFrame*>(out), false, true, 1023);
            break;
        default:
            break;
        }
    }

    return S_OK;
}

winrt::hresult NVFrame::RegisterD3D11Resource(ID3D11Resource* resource)
{
    try {
        winrt::check_pointer(resource);
        winrt::check_hresult(CUDAHr(cudaGraphicsD3D11RegisterResource(&cuResource, resource, cudaGraphicsRegisterFlagsNone)));
        winrt::check_hresult(CUDAHr(cudaGraphicsMapResources(1, &cuResource)));
        winrt::check_hresult(CUDAHr(cudaGraphicsSubResourceGetMappedArray(&resourceArray, cuResource, 0, 0)));
    }catch (winrt::hresult_error const& err) {
        return err.code();
    }

    return S_OK;
}

winrt::hresult NVFrame::CommitResource()
{
    try {
        winrt::check_hresult(CUDAHr(cudaMemcpy2DToArray(resourceArray, 0, 0, (const void*)cuFrame,
            cuDesc.width * GetChannelFactor(cuDesc.format),
            cuDesc.width * GetChannelFactor(cuDesc.format), cuDesc.height,
            cudaMemcpyDeviceToDevice)));
    } catch (winrt::hresult_error const& err) {
        return err.code();
    }
    
    return S_OK;
}

winrt::hresult NVFrame::ConfigureFrame(FRAME_OUTPUT_DESC desc)
{
    if (desc.width == 0 || desc.height == 0)
        return E_INVALIDARG;
    if (desc.format < HV_FORMAT_RGBA8 || desc.format > HV_FORMAT_P016)
        return E_INVALIDARG;

    try {
        winrt::check_hresult(CUDAHr(cudaMalloc((void**)&cuFrame, GetChannelFactor(desc.format) * desc.width * desc.height)));
        winrt::check_pointer(&cuFrame);
    }catch (winrt::hresult_error const& err) {
        return err.code();
    }

    return S_OK;
}
