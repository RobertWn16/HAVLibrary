#include "WinDisplay.hpp"

winrt::hresult WinDisplay::ConfigureDisplay() noexcept
{
    try {
        winrt::com_ptr<IDXGIFactory1> pwdFactory;
        winrt::com_ptr<IDXGIAdapter> pwdAdapter;
        winrt::com_ptr<IDXGIDevice> pwdDxgiDevice;
        winrt::com_ptr<IDXGIOutput> pwdOutput;
        DXGI_OUTPUT_DESC1 output_desc1 = { 0 };

        winrt::check_hresult(CreateDXGIFactory1(IID_IDXGIFactory1, pwdFactory.put_void()));
        winrt::check_hresult(pwdFactory->EnumAdapters(0, pwdAdapter.put()));
        winrt::check_hresult(D3D11CreateDevice(pwdAdapter.get(), D3D_DRIVER_TYPE_UNKNOWN,
            nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, pwdDevice.put(), nullptr, pwdDeviceCtx.put()));
        winrt::check_hresult(pwdAdapter->EnumOutputs(0, pwdOutput.put()));
        pwdOutput.as(pwdOutput6);
        winrt::check_pointer(pwdOutput6.get());
        pwdOutput6->GetDesc1(&output_desc1);

        display_desc.width = output_desc1.DesktopCoordinates.right - output_desc1.DesktopCoordinates.left;
        display_desc.heigth = output_desc1.DesktopCoordinates.bottom - output_desc1.DesktopCoordinates.top;
        display_desc.bitdepth = output_desc1.BitsPerColor;
        display_desc.colorspace = DXGICsHV(output_desc1.ColorSpace);
        display_desc.colorimetry.xr = output_desc1.RedPrimary[0];
        display_desc.colorimetry.yr = output_desc1.RedPrimary[1];
        display_desc.colorimetry.xr = output_desc1.GreenPrimary[0];
        display_desc.colorimetry.yr = output_desc1.GreenPrimary[1];
        display_desc.colorimetry.xr = output_desc1.BluePrimary[0];
        display_desc.colorimetry.yr = output_desc1.BluePrimary[1];
        display_desc.max_display_luminance = output_desc1.MaxLuminance;
        display_desc.avg_display_luminance = output_desc1.MaxFullFrameLuminance;

    }catch (winrt::hresult_error const& err) {
        return err.code();
    }

    return S_OK;
}

winrt::hresult WinDisplay::DisplayCapture(IVideoSource** out) noexcept
{
    winrt::com_ptr<WinDisplayVideoSource> pwdVideoSource;
    IDXGIOutputDuplication *pwdOutputDuplication;
    VIDEO_SOURCE_DESC vsrc_desc = { 0 };

    try {
        winrt::check_hresult(pwdOutput6->DuplicateOutput(pwdDevice.get(), &pwdOutputDuplication));
        pwdVideoSource = winrt::make_self<WinDisplayVideoSource>();
        winrt::check_pointer(pwdVideoSource.get());

        vsrc_desc.width = display_desc.width;
        vsrc_desc.heigth = display_desc.heigth;
        vsrc_desc.bitdepth = display_desc.bitdepth;
        vsrc_desc.avg_content_luminance = display_desc.avg_display_luminance;
        vsrc_desc.format = DXGIFmtHV(DXGI_FORMAT_B8G8R8A8_UNORM);

        pwdVideoSource->ConfigureVideoSource(vsrc_desc, pwdDevice.get(), pwdOutputDuplication);
        *out = pwdVideoSource.detach();
    }catch (winrt::hresult_error const& err) {
        return err.code();
    }
    return winrt::hresult();
}

winrt::hresult WinDisplay::GetDesc(DISPLAY_DESC& desc)
{
    desc = display_desc;
    return S_OK;
}
