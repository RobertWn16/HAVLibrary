#include "DevNVIDIA.hpp"

winrt::hresult DevNVIDIA::InitDevice(DEV_DESC dev_desc)
{
    try
    {
        winrt::check_hresult(cuInit(0));
        cuDeviceGet(&cuDevice, nv_desc.ordinal);
        cuCtxCreate(&cuContext, 0, cuDevice);
        nv_desc = dev_desc;
    }catch (winrt::hresult_error const& err) {
        return err.code();
    }

    return S_OK;
}

winrt::hresult DevNVIDIA::GetDesc(DEV_DESC& desc)
{
    return E_FAIL;
}

CUdevice DevNVIDIA::GetDevice()
{
    return cuDevice;
}

CUcontext DevNVIDIA::GetContext()
{
    return cuContext;
}
