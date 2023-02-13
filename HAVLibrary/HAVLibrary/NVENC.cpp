#include "NVENC.hpp"

#pragma comment (lib, "nvencodeapi")
static winrt::hresult NVENCStatHr(NVENCSTATUS nvenc_stat)
{
    if (nvenc_stat != NV_ENC_SUCCESS)
        return E_INVALIDARG;
    return S_OK;
}
winrt::hresult NVENC::IsSupported(VIDEO_SOURCE_DESC desc)
{
    return winrt::hresult();
}

winrt::hresult NVENC::Encode(IFrame* out)
{
    return winrt::hresult();
}

winrt::hresult NVENC::ConfigureEncoder(CUcontext deviceContext)
{
    try
    {
        NV_ENCODE_API_FUNCTION_LIST functionList = { NV_ENCODE_API_FUNCTION_LIST_VER };
        NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS nvencOpenPar = { NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER };
        winrt::check_pointer(&deviceContext);
        winrt::check_hresult(NVENCStatHr(NvEncodeAPICreateInstance(&functionList)));
        nvencOpenPar.apiVersion = NVENCAPI_VERSION;
        nvencOpenPar.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
        nvencOpenPar.device = deviceContext;
        winrt::check_hresult(NVENCStatHr(functionList.nvEncOpenEncodeSessionEx(&nvencOpenPar, &nvencEncoder)));
    }catch (winrt::hresult_error& err) {
        return err.code();
    }
    return S_OK;
}
