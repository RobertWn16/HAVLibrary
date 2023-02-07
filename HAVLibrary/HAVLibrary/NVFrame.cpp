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
                    hav_p016_HDR10_bgra64_HDR10_PQ_ACES(np_16u_Image[0], np_16u_Image[1], cuDesc.width, cuDesc.height, false, content_Colorimetry_XYZ,
                        nv_out->display_Colrimetry_XYZ_Inverse, cuDesc.max_content_luminance, cuDesc.display_luminance, reinterpret_cast<unsigned short*>(nv_out->cuFrame), true, 0.0f);

                if (cuDesc.tone_mapper == HV_TONE_MAPPER_REINHARD_EXT)
                    hav_p016_HDR10_bgra64_HDR10_PQ_Reinhard(np_16u_Image[0], np_16u_Image[1], cuDesc.width, cuDesc.height, false, content_Colorimetry_XYZ, 
                        nv_out->display_Colrimetry_XYZ_Inverse, cuDesc.max_content_luminance, cuDesc.display_luminance, reinterpret_cast<unsigned short*>(nv_out->cuFrame), true, 0.0f);

            }

            if (cuDesc.transfer == HV_TRANSFER_LINEAR) {
                hav_p016_HDR10_bgra64_HDR10_Linear(np_16u_Image[0], np_16u_Image[1],
                    cuDesc.width, cuDesc.height, false, cuDesc.display_luminance, cuDesc.wr, cuDesc.wb, reinterpret_cast<unsigned short*>(nv_out->cuFrame), true, 0.0f);
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
        if (desc.content_colorspace == HV_COLORSPACE_UNKNOWN)
            return E_INVALIDARG;
        if (desc.width < 0 || desc.height < 0)
            return E_INVALIDARG;
        if (desc.format == HV_FORMAT_UNKNOWN)
            return E_INVALIDARG;

        winrt::check_hresult(CUDAHr(cudaMalloc((void**)&cuFrame, GetChannelFactor(desc.format) * desc.width * desc.height)));
        winrt::check_pointer(&cuFrame);

        cuDesc = desc;
        switch (cuDesc.content_colorspace)
        {
        case HV_COLORSPACE_BT601:
            cuDesc.content_colorimetry = BT601_Colorimetry_625lines;
            break;
        case HV_COLORSPACE_BT709:
            cuDesc.content_colorimetry = BT709_Colorimetry;
            break;
        case HV_COLORSPACE_BT2020:
            cuDesc.content_colorimetry = BT2020_Colorimetry;
            break;
        case HV_COLORSPACE_DISPLAY_P3:
            cuDesc.content_colorimetry = DisplayP3_Colorimetry;
            break;
        default:
            break;
        }

        switch (cuDesc.display_colorspace)
        {
        case HV_COLORSPACE_BT601:
            cuDesc.display_colorimetry = BT601_Colorimetry_625lines;
            break;
        case HV_COLORSPACE_BT709:
            cuDesc.display_colorimetry = BT709_Colorimetry;
            break;
        case HV_COLORSPACE_BT2020:
            cuDesc.display_colorimetry = BT2020_Colorimetry;
            break;
        case HV_COLORSPACE_DISPLAY_P3:
            cuDesc.display_colorimetry = DisplayP3_Colorimetry;
            break;
        default:
            break;
        }

        ComputeColorimetryMatrix(cuDesc.content_colorimetry, content_Colorimetry_XYZ, content_Colorimetry_XYZ_Inverse);
        ComputeColorimetryMatrix(cuDesc.display_colorimetry, display_Colorimetry_XYZ, display_Colrimetry_XYZ_Inverse);

    }catch (winrt::hresult_error const& err) {
        return err.code();
    }

    return S_OK;
}

winrt::hresult NVFrame::ComputeColorimetryMatrix(HVColorimetry in_colorimetry, float XYZ[3][3], float XYZ_Inverse[3][3])
{
    float ref_white_inv[3][3];
    float ref_white[3][3];
    float ref_white_cord[3];
    float transpose_inv[3][3];
    float wr = 0.0f, wg = 0.0f, wb = 0.0f;

    ref_white[0][0] = in_colorimetry.xr / in_colorimetry.yr;
    ref_white[1][0] = 1;
    ref_white[2][0] = (1 - in_colorimetry.xr - in_colorimetry.yr) / in_colorimetry.yr;

    ref_white[0][1] = in_colorimetry.xg / in_colorimetry.yg;
    ref_white[1][1] = 1;
    ref_white[2][1] = (1 - in_colorimetry.xg - in_colorimetry.yg) / in_colorimetry.yg;

    ref_white[0][2] = in_colorimetry.xb / in_colorimetry.yb;
    ref_white[1][2] = 1;
    ref_white[2][2] = (1 - in_colorimetry.xb - in_colorimetry.yb) / in_colorimetry.yb;
    
    ref_white_cord[0] = in_colorimetry.xw / in_colorimetry.yw;
    ref_white_cord[1] = 1;
    ref_white_cord[2] = (1 - in_colorimetry.xw - in_colorimetry.yw) / in_colorimetry.yw;
    ComputeMatrixInverse(ref_white, ref_white_inv);

    wr = ref_white_inv[0][0] * ref_white_cord[0] + ref_white_inv[1][0] * ref_white_cord[1] + ref_white_inv[2][0] * ref_white_cord[2];
    wg = ref_white_inv[0][1] * ref_white_cord[0] + ref_white_inv[1][1] * ref_white_cord[1] + ref_white_inv[2][1] * ref_white_cord[2];
    wb = ref_white_inv[0][2] * ref_white_cord[0] + ref_white_inv[1][2] * ref_white_cord[1] + ref_white_inv[2][2] * ref_white_cord[2];

    for (int i = 0; i < 3; i++) {
        XYZ[i][0] = wr * ref_white[i][0];
        XYZ[i][1] = wg * ref_white[i][1];
        XYZ[i][2] = wb * ref_white[i][2];
    }

    ComputeMatrixInverse(XYZ, transpose_inv);

    for (int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            XYZ_Inverse[i][j] = transpose_inv[j][i];
    return S_OK;
}

void NVFrame::ComputeMatrixInverse(float mat[3][3], float mat_inv[3][3])
{
    float determinant = 0.0f;
    for (int i = 0; i < 3; i++)
        determinant = determinant + (mat[0][i] * (mat[1][(i + 1) % 3] * mat[2][(i + 2) % 3] - mat[1][(i + 2) % 3] * mat[2][(i + 1) % 3]));

    if (determinant < 0.001f)
        return;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
           mat_inv[i][j] = ((mat[(i + 1) % 3][(j + 1) % 3] * mat[(i + 2) % 3][(j + 2) % 3]) - (mat[(i + 1) % 3][(j + 2) % 3] * mat[(i + 2) % 3][(j + 1) % 3])) / determinant;
    }
}
