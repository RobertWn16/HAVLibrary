#include "DevNVIDIA.hpp"

winrt::hresult DevNVIDIA::InitDevice(DEV_DESC dev_desc)
{
    nv_desc = dev_desc;
    winrt::com_ptr<IDXGIFactory> dxgi_fty;
    winrt::check_hresult(CreateDXGIFactory(IID_IDXGIFactory, dxgi_fty.put_void()));

    int ite = 0;
    int nvidia_adpt = 0;
    IDXGIAdapter* dxgi_adpt;
    DXGI_ADAPTER_DESC desc;

    while (dxgi_fty->EnumAdapters(ite, &dxgi_adpt) == S_OK)
    {
        dxgi_adpt->GetDesc(&desc);
        if (desc.VendorId == nv_desc.vendor) {
            if (nvidia_adpt == nv_desc.ordinal) {

                winrt::check_hresult(cuInit(0));
                cuDeviceGet(&cuDevice, nv_desc.ordinal);
                cuCtxCreate(&cuContext, 0, cuDevice);

                dxgi_adpt->Release();
                break;
            }
            nvidia_adpt++;
        }
        dxgi_adpt->Release();
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
