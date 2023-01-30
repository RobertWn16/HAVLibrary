#include "HAV.hpp"
#include <cuda_fp16.h>
#include <dxgi1_6.h>
#pragma comment (lib, "d3d11")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dxguid.lib")
D3D_FEATURE_LEVEL levels[] = {
D3D_FEATURE_LEVEL_11_0,
D3D_FEATURE_LEVEL_11_1
};
ID3D11Device* pd3d11_dev = nullptr;
ID3D11DeviceContext* pd3d11_ctx = nullptr;
IDXGIFactory2* pdxgi_fty = nullptr;
IDXGIAdapter* pdxgi_adpt = nullptr;
IDXGIDevice3* pdxgi_dev = nullptr;
winrt::com_ptr<IDemuxer> demx;
winrt::com_ptr<HAV> hav_instance = winrt::make_self<HAV>();
winrt::com_ptr<IDev> dev_nvidia;

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_CLOSE:
        DestroyWindow(hwnd);
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hwnd, msg, wParam, lParam);
    }
    return 0;
}
struct THREAD_PARAMS
{
    HWND hwnd;
    std::string name;
};
void DecodeMainLoop(THREAD_PARAMS par)
{
    HRESULT hr = S_OK;
    D3D11_VIEWPORT m_viewport = {  };
    D3D11_TEXTURE2D_DESC bkbuffer_desc = { };
    D3D_FEATURE_LEVEL d3d_clevel;

    ID3D11Texture2D* pdxgi_resourced_texture = nullptr;
    ID3D11Texture2D* pd3d11_bkbuffer = nullptr;
    ID3D11Texture2D* pd3d11_cuda_shresource = nullptr;

    //DXGISWPCH Garbage
    DXGI_SWAP_CHAIN_DESC1 dxgi_desc = { };
    DXGI_SWAP_CHAIN_FULLSCREEN_DESC dxgi_desc_fs = { };

    IDXGISwapChain1* pdxgi_swpch = nullptr;


    if (SUCCEEDED(hr))
    {
        dxgi_desc.BufferCount = 3;
        dxgi_desc.Format = DXGI_FORMAT_R10G10B10A2_UNORM;
        dxgi_desc.BufferUsage = DXGI_USAGE_BACK_BUFFER;
        dxgi_desc.SampleDesc.Count = 1;      //multisampling setting
        dxgi_desc.SampleDesc.Quality = 0;    //vendor-specific flag
        dxgi_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
        dxgi_desc.Scaling = DXGI_SCALING_STRETCH;

        dxgi_desc_fs.RefreshRate.Numerator = 1;
        dxgi_desc_fs.RefreshRate.Denominator = 60;
        dxgi_desc_fs.Scaling = DXGI_MODE_SCALING_STRETCHED;
        dxgi_desc_fs.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_PROGRESSIVE;
        dxgi_desc_fs.Windowed = true;
        
        hr = pdxgi_fty->CreateSwapChainForHwnd(pd3d11_dev, par.hwnd, &dxgi_desc, NULL, NULL, &pdxgi_swpch);
        DXGI_RGBA rgba = { };
        rgba.a = 255;
        pdxgi_swpch->SetBackgroundColor(&rgba);
    }

    winrt::com_ptr<IVideoSource> source;
    winrt::com_ptr<IDecoder> dec;
    winrt::com_ptr<IFrame> nv_frame;
    winrt::com_ptr<IFrame> rgba_frame;

    VIDEO_SOURCE_DESC vsrc_desc = { 0 };
    try {
        winrt::check_hresult(hav_instance->CreateDecoder(IID_HAV_NVDEC, dec.put()));
        winrt::check_hresult(demx->VideoCapture(par.name.c_str(), source.put()));
        winrt::check_hresult(hav_instance->Link(dev_nvidia.get(), dec.get()));

        winrt::check_hresult(source->GetDesc(vsrc_desc));
        winrt::check_hresult(dec->IsSupported(vsrc_desc));
        winrt::check_hresult(hav_instance->Link(source.get(), dec.get()));

        FRAME_OUTPUT_DESC frame_desc = {};
        frame_desc.format = vsrc_desc.format;
        frame_desc.width = vsrc_desc.width;
        frame_desc.height = vsrc_desc.heigth;
        winrt::check_hresult(hav_instance->CreateFrame(IID_HAV_NVFrame, frame_desc, nv_frame.put()));

        frame_desc.format = HV_FORMAT_BGRA64_HDR10;
        winrt::check_hresult(hav_instance->CreateFrame(IID_HAV_NVFrame, frame_desc, rgba_frame.put()));

        hr = pdxgi_swpch->ResizeBuffers(3, vsrc_desc.width, vsrc_desc.heigth, DXGI_FORMAT_R16G16B16A16_FLOAT, 0);
    }
    catch (winrt::hresult_error const& err) {
        std::cout << "Error 0x%x" << err.code();
        return;
    }

    if (SUCCEEDED(hr))
    {
        hr = pdxgi_swpch->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&pd3d11_bkbuffer);

        D3D11_TEXTURE2D_DESC tex_bkbuffer;
        pd3d11_bkbuffer->GetDesc(&tex_bkbuffer);

        pd3d11_bkbuffer->Release();

        IDXGISwapChain3* spwch;
        hr = pdxgi_swpch->QueryInterface(&spwch);
        hr = spwch->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709);
        D3D11_TEXTURE2D_DESC tex_desc = { 0 };
        tex_desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        tex_desc.ArraySize = 1;
        tex_desc.MipLevels = 1;
        tex_desc.SampleDesc.Count = 1;
        tex_desc.Width = vsrc_desc.width;
        tex_desc.Height = vsrc_desc.heigth;
        tex_desc.BindFlags = D3D11_BIND_RENDER_TARGET;
        tex_desc.Usage = D3D11_USAGE_DEFAULT;

        hr = pd3d11_dev->CreateTexture2D(&tex_bkbuffer, nullptr, &pd3d11_cuda_shresource);

        pd3d11_bkbuffer->Release();
    }

    winrt::check_hresult(rgba_frame->RegisterD3D11Resource(pd3d11_cuda_shresource));

    while (true) {
        try {
            winrt::check_hresult(dec->Decode(nv_frame.get()));
            winrt::check_hresult(nv_frame->ConvertFormat(HV_FORMAT_RGBA8, rgba_frame.get()));
            winrt::check_hresult(rgba_frame->CommitResource());

           if (SUCCEEDED(hr)) hr = pdxgi_swpch->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&pd3d11_bkbuffer);
            pd3d11_ctx->CopyResource(pd3d11_bkbuffer, pd3d11_cuda_shresource);
            pdxgi_swpch->Present(0, 0);
            pd3d11_bkbuffer->Release();

        }
        catch (winrt::hresult_error const& err) {

        }
    }
}


int main(int argc, char** argv)
{
    WNDCLASSEX wc;
    HWND hwnd;
    MSG Msg;

    //Step 1: Registering the Window Class
    wc.cbSize = sizeof(WNDCLASSEX);
    wc.style = 0;
    wc.lpfnWndProc = WndProc;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = GetModuleHandle(NULL);
    wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszMenuName = NULL;
    wc.lpszClassName = L"DLSSClass";
    wc.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

    if (!RegisterClassEx(&wc))
    {
        return 0;
    }

    // Step 2: Creating the Window
    hwnd = CreateWindowEx(
        WS_EX_CLIENTEDGE,
        L"DLSSClass",
        L"The title of my window",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 1600, 763,
        NULL, NULL, GetModuleHandle(NULL), NULL);

    if (hwnd == NULL)
    {
        return 0;
    }

    HWND core_hwnd = CreateWindowEx(
        WS_EX_CLIENTEDGE,
        L"DLSSClass",
        L"Second",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 1600, 763,
        NULL, NULL, GetModuleHandle(NULL), NULL);

    HRESULT hr = S_OK;
    if (SUCCEEDED(hr)) hr = CreateDXGIFactory1(IID_IDXGIFactory3, (void**)&pdxgi_fty);

    if (SUCCEEDED(hr)) hr = pdxgi_fty->EnumAdapters(0, (IDXGIAdapter**)&pdxgi_adpt);

    if (SUCCEEDED(hr))
    {
        hr = D3D11CreateDevice(
            pdxgi_adpt,
            D3D_DRIVER_TYPE_UNKNOWN,
            nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            levels,
            2,
            D3D11_SDK_VERSION,
            &pd3d11_dev,
            nullptr,
            &pd3d11_ctx
        );
    }

    if (SUCCEEDED(hr))
    {
        pdxgi_adpt->Release();
        pdxgi_adpt = nullptr;
        pdxgi_fty->Release();
        pdxgi_fty = nullptr;
    }

    if (SUCCEEDED(hr)) hr = pd3d11_dev->QueryInterface(__uuidof(IDXGIDevice3), (void**)&pdxgi_dev);

    if (SUCCEEDED(hr)) hr = pdxgi_dev->GetAdapter(&pdxgi_adpt);

    if (SUCCEEDED(hr)) hr = pdxgi_adpt->GetParent(__uuidof(IDXGIFactory2), (void**)&pdxgi_fty);

    ShowWindow(core_hwnd, SW_SHOWDEFAULT);
    UpdateWindow(core_hwnd);

    DEV_DESC dev_desc;
    dev_desc.vendor = NVIDIA;
    dev_desc.ordinal = 0;

    winrt::check_hresult(hav_instance->CreateDevice(IID_HAV_NVDev, dev_desc, dev_nvidia.put()));
    winrt::check_hresult(hav_instance->CreateDemuxer(IID_HAV_FFMPEGDemuxer, demx.put()));
    THREAD_PARAMS par;

    par.hwnd = core_hwnd;
    par.name = "sonycolor.mkv";
    CreateThread(nullptr,
        0,
        reinterpret_cast<LPTHREAD_START_ROUTINE>(DecodeMainLoop),
        &par,
        0,
        nullptr);


    while (GetMessageW(&Msg, nullptr, 0, 0))
    {
        TranslateMessage(&Msg);
        DispatchMessage(&Msg);
    }
}