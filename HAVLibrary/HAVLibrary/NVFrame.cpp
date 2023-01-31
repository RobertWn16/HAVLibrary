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

    NVFrame* nv_out = dynamic_cast<NVFrame*>(out);
    if (!nv_out)
        return E_INVALIDARG;

    unsigned short* np_16u_Image[2];
    unsigned char* np_8u_Image[2];
    if (cuDesc.format == HV_FORMAT_NV12) {
        switch (out_frame_desc.format)
        {
        case HV_FORMAT_BGRA32:
            np_8u_Image[0] = reinterpret_cast<unsigned char*>(cuFrame);
            np_8u_Image[1] = reinterpret_cast<unsigned char*>(cuFrame) + cuDesc.width * cuDesc.height;
            hav_nv12_bgra32_SDR(np_8u_Image[0], np_8u_Image[1], cuDesc.width, cuDesc.height, false, 
                cuDesc.wr, cuDesc.wb, reinterpret_cast<unsigned char*>(nv_out->cuFrame), true, 255.0f);
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
            np_16u_Image[0] = reinterpret_cast<unsigned short*>(cuFrame);
            np_16u_Image[1] = reinterpret_cast<unsigned short*>(cuFrame) + cuDesc.width * cuDesc.height;
            hav_p016_HDR10_bgra32_SDR_Linear(np_16u_Image[0], np_16u_Image[1], cuDesc.width, cuDesc.height,
                false, cuDesc.wr, cuDesc.wb, reinterpret_cast<unsigned char*>(nv_out->cuFrame), true, 0.0f);
            break;
        case HV_FORMAT_BGRA64_HDR10:
            np_16u_Image[0] = reinterpret_cast<unsigned short*>(cuFrame);
            np_16u_Image[1] = reinterpret_cast<unsigned short*>(cuFrame) + cuDesc.width * cuDesc.height;
            if (cuDesc.transfer == HV_TRANSFER_PQ) {
                if (cuDesc.tone_mapper == HV_TONE_MAPPER_ACES)
                    hav_p016_HDR10_bgra64_HDR10_PQ_ACES(np_16u_Image[0], np_16u_Image[1],
                        cuDesc.width, cuDesc.height, false, cuDesc.max_content_luminance, cuDesc.display_luminance,
                        cuDesc.wr, cuDesc.wb, reinterpret_cast<unsigned short*>(nv_out->cuFrame), true, 0.0f);

                if (cuDesc.tone_mapper == HV_TONE_MAPPER_REINHARD_EXT)
                    hav_p016_HDR10_bgra64_HDR10_PQ_Reinhard(np_16u_Image[0], np_16u_Image[1],
                        cuDesc.width, cuDesc.height, false, cuDesc.max_content_luminance, cuDesc.display_luminance,
                        cuDesc.wr, cuDesc.wb, reinterpret_cast<unsigned short*>(nv_out->cuFrame), true, 0.0f);

            }
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

        cuDesc.colorspace = desc.colorspace;
        switch (desc.colorspace)
        {
        case HV_COLORSPACE_BT601:
            cuDesc.wr = 0.299f; cuDesc.wb = 0.114f;
            break;
        case HV_COLORSPACE_BT709:
            cuDesc.wr = 0.2126f; cuDesc.wr = 0.0722f;
            break;
        case HV_COLORSPACE_BT2020:
            cuDesc.wr = 0.2627f; cuDesc.wb = 0.0593f;
            break;
        case HV_COLORSPACE_CUSTOM:
            cuDesc.wr = desc.wr; cuDesc.wb = desc.wb;
            break;
        default:
            break;
        }
    }catch (winrt::hresult_error const& err) {
        return err.code();
    }

    return S_OK;
}
