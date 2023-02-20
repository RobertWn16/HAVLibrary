#include "HAV.hpp"
#include <cuda_fp16.h>
#include <dxgi1_6.h>
#include "FFMPEGMuxer.hpp"
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
winrt::com_ptr<IVideoSource> display_source;

winrt::com_ptr<IPacket> pckt;
winrt::com_ptr<IEncoder> encoder;
winrt::com_ptr<FFMPEGMuxer> ffmpeg_muxer;
winrt::com_ptr<IVideoOutput> vout;

struct THREAD_PARAMS
{
    HWND hwnd;
    std::string name;
    bool windowIsClosed = false;
};

void DesktopDuplication(THREAD_PARAMS par)
{
    winrt::com_ptr<IFrame> nv_frame;
    IDXGISwapChain1* pdxgi_swpch = nullptr;
    HRESULT hr = S_OK;
    ID3D11Texture2D* out_tex = nullptr;
    DXGI_SWAP_CHAIN_DESC1 dxgi_desc = { };
    DXGI_SWAP_CHAIN_FULLSCREEN_DESC dxgi_desc_fs = { };

    FRAME_OUTPUT_DESC frame_desc = {};
    frame_desc.format = HV_FORMAT_BGRA32;
    frame_desc.width = 3840;
    frame_desc.height = 2160;
    frame_desc.content_colorspace = HV_COLORSPACE_BT2020;
    frame_desc.display_colorspace = HV_COLORSPACE_DISPLAY_P3;

    winrt::check_hresult(hav_instance->CreateFrame(IID_HAV_NVFrame, frame_desc, nv_frame.put()));

    display_source->Parse(&out_tex);
    nv_frame->RegisterD3D11Resource(out_tex);

    ENCODER_DESC nvenc_desc;
    nvenc_desc.codec = HV_CODEC_H264;
    nvenc_desc.encoded_height = 2160;
    nvenc_desc.encoded_width = 3840;
    nvenc_desc.max_encoded_height = 3840;
    nvenc_desc.max_encoded_width = 2160;
    nvenc_desc.preset = HV_ENCODER_LOW_QUALITY_H264;
    nvenc_desc.framerate_num = 60;
    nvenc_desc.framerate_den = 1;
    nvenc_desc.num_in_back_buffers = 1;
    nvenc_desc.num_out_back_buffers = 2;

    winrt::check_hresult(hav_instance->CreateEncoder(IID_HAV_NVENC, nvenc_desc, dev_nvidia.get(), encoder.put()));
    hav_instance->CreatePacket(IID_HAV_FFMPEGPacket, pckt.put());


    VIDEO_OUTPUT_DESC vout_desc;
    vout_desc.codec = HV_CODEC_H264;
    vout_desc.container = HV_CONTAINER_MP4;
    vout_desc.width = 3840;
    vout_desc.height = 2160;
    ffmpeg_muxer->VideoStream("tcp://127.0.0.1:8080/live.h264?listen", vout_desc, vout.put());

    SYSTEMTIME start, stop;
    while (!par.windowIsClosed) {
        try {
            display_source->Parse(&out_tex);
            nv_frame->CommitFrame();
            hr = encoder->Encode(nv_frame.get());
            if (FAILED(hr)) {
                break;
            }
        }
        catch (winrt::hresult_error const& err) {
            std::cout << err.code();
        }
    }
}

int main(int argc, char** argv)
{
  
    DEV_DESC dev_desc;
    dev_desc.vendor = NVIDIA;
    dev_desc.ordinal = 0;

    winrt::com_ptr<IDisplay> display;
    ffmpeg_muxer = winrt::make_self<FFMPEGMuxer>();
    winrt::check_hresult(hav_instance->CreateDevice(IID_HAV_NVDev, dev_desc, dev_nvidia.put()));
    winrt::check_hresult(hav_instance->CreateDemuxer(IID_HAV_FFMPEGDemuxer, demx.put()));
    winrt::check_hresult(hav_instance->CreateDisplay(IID_HAV_WinDisplay, display.put()));

    display->DisplayCapture(display_source.put());
    THREAD_PARAMS par;

    par.windowIsClosed = false;
    HANDLE hThread = CreateThread(nullptr,
        0,
        reinterpret_cast<LPTHREAD_START_ROUTINE>(DesktopDuplication),
        &par,
        0,
        nullptr);

    
    HRESULT hr = S_OK;
    int i = 0;
    while (!par.windowIsClosed) {
        if (pckt.get())
        {
            hr = encoder->GetEncodedPacket(pckt.get());
            if (SUCCEEDED(hr))
                hr = vout->Write(pckt.get());
        }
    }

    par.windowIsClosed = true;
    WaitForSingleObject(hThread, INFINITE);
}