#include "WinDisplayVideoSource.hpp"

constexpr int MAX_TIMEOUT = 100;

winrt::hresult WinDisplayVideoSource::GetDesc(VIDEO_SOURCE_DESC& desc)
{
    desc = video_source_desc;
    return S_OK;
}

winrt::hresult WinDisplayVideoSource::Parse(void* desc)
{
    return winrt::hresult();
}
SYSTEMTIME start, stop;
winrt::hresult WinDisplayVideoSource::Parse(ID3D11Texture2D **Out) noexcept
{
    DXGI_OUTDUPL_FRAME_INFO outdpl_frame_info = { 0 };
    winrt::com_ptr<IDXGIResource> pwdsDxgiResource;
    winrt::com_ptr<ID3D11Texture2D> pwdsBufTex;
    D3D11_TEXTURE2D_DESC desc;
    HRESULT hr = S_OK;

    hr = pwdsOutputDupl->AcquireNextFrame(400, &outdpl_frame_info, pwdsDxgiResource.put());
    switch (hr)
    {
    case DXGI_ERROR_ACCESS_LOST:
        pwdsOutputDupl.detach()->Release();
        hr = pwdsOutput6->DuplicateOutput(pwdsDevice, pwdsOutputDupl.put());
        return hr;
        break;
    case DXGI_ERROR_WAIT_TIMEOUT:
        return ERROR_TIMEOUT;
        break;
    default:
        break;
    }

    try {
        winrt::check_pointer(pwdsDxgiResource.get());
        pwdsDxgiResource.as(pwdsBufTex);
        pwdsBufTex->GetDesc(&desc);
        winrt::check_pointer(pwdsBufTex.get());
        pwdsDeviceCtx->CopyResource(pwdsTex.get(), pwdsBufTex.get());
        *Out = pwdsTex.get();
        pwdsOutputDupl->ReleaseFrame();
    }catch (winrt::hresult_error const& err) {
        return err.code();
    }

    return S_OK;
}

winrt::hresult WinDisplayVideoSource::ConfigureVideoSource(VIDEO_SOURCE_DESC vsrc_desc, ID3D11Device* pDevice, IDXGIOutput6* pdxgiOutput)
{

    winrt::com_ptr<IDXGIDevice> pwdsDxgiDevice;
    IDXGIOutputDuplication* pdxgiOutputDupl = nullptr;
    try {  
        winrt::check_pointer(pdxgiOutput);
        pwdsOutput6 = pdxgiOutput;
        winrt::check_pointer(pDevice);
        pwdsDevice = pDevice;
        pwdsDevice->GetImmediateContext(&pwdsDeviceCtx);
        winrt::check_pointer(&pwdsDeviceCtx);
        std::cout << "Starting duplication" << std::endl;
        winrt::check_pointer(pdxgiOutput);
        winrt::check_hresult(pdxgiOutput->DuplicateOutput(pDevice, &pdxgiOutputDupl));
        pwdsOutputDupl.attach(pdxgiOutputDupl);
        video_source_desc = vsrc_desc;

        std::cout << "Succedeed duplication. Width of stream " << video_source_desc.width << std::endl;
        DXGI_OUTDUPL_DESC outdupl_desc = {0};
        pwdsOutputDupl->GetDesc(&outdupl_desc);
        video_source_desc.framerate = (double)(outdupl_desc.ModeDesc.RefreshRate.Numerator) / outdupl_desc.ModeDesc.RefreshRate.Denominator;


        D3D11_TEXTURE2D_DESC tex_desc = {0};
        tex_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        tex_desc.ArraySize = 1;
        tex_desc.MipLevels = 1;
        tex_desc.SampleDesc.Count = 1;
        tex_desc.Width = video_source_desc.width;
        tex_desc.Height = video_source_desc.heigth;
        tex_desc.BindFlags = D3D11_BIND_RENDER_TARGET;
        tex_desc.Usage = D3D11_USAGE_DEFAULT;
        winrt::check_hresult(pwdsDevice->CreateTexture2D(&tex_desc, nullptr, pwdsTex.put()));


    }catch (winrt::hresult_error const& err) {
        std::cout << std::hex << err.code();
    }

    return S_OK;
}
