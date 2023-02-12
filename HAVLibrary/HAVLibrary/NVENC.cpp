#include "NVENC.hpp"

winrt::hresult NVENC::IsSupported(VIDEO_SOURCE_DESC desc)
{
    return winrt::hresult();
}

winrt::hresult NVENC::Encode(IFrame* out)
{
    return winrt::hresult();
}

winrt::hresult NVENC::ConfigureEncoder()
{
    CUdevice cuDevice;
    cuDeviceGet(&cuDevice, 0);
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS nvencOpenPar;
    nvencOpenPar.apiVersion = NVENCAPI_VERSION;
    nvencOpenPar.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    nvencOpenPar.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    nvencOpenPar.device = &cuDevice;
    NVENCSTATUS stats = NvEncOpenEncodeSessionEx(&nvencOpenPar, &nvencEncoder);
    return winrt::hresult();
}
