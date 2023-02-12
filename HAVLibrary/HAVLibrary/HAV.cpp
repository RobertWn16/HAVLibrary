#include "HAV.hpp"

winrt::hresult HAV::Link(IHAVComponent* In, IHAVComponent* Out)
{
    DevNVIDIA* dev_nvidia = dynamic_cast<DevNVIDIA*>(In);
    if (dev_nvidia) {
        NVDEC* nv_decoder = dynamic_cast<NVDEC*>(Out);
        if (nv_decoder) {
            if (!nv_decoder->hasDevice) {
                nv_decoder->deviceContext = dev_nvidia->GetContext();
                nv_decoder->hasDevice = true;
                return S_OK;
            }
            else
                return E_HV_ALREADY_LINKED;
        }
        return E_INVALIDARG;
    }

    FFMPEGVideoSource* ffmpeg_source = dynamic_cast<FFMPEGVideoSource*>(In);
    if (ffmpeg_source)
    {
        NVDEC* nv_decoder = dynamic_cast<NVDEC*>(Out);
        if (nv_decoder) {
            if (!nv_decoder->hasSource) {
                winrt::check_hresult(nv_decoder->IsSupported(ffmpeg_source->source_desc));
                winrt::check_hresult(nv_decoder->CreateParser(ffmpeg_source->source_desc));
                nv_decoder->vSource = ffmpeg_source;
                nv_decoder->hasSource = true;
                return S_OK;
            }
            else
                return E_HV_ALREADY_LINKED;
        }
        NVJpegDecoder* nvjpeg_decoder = dynamic_cast<NVJpegDecoder*>(Out);
        if (nvjpeg_decoder) {
                winrt::check_hresult(nvjpeg_decoder->IsSupported(ffmpeg_source->source_desc));
                winrt::check_hresult(nvjpeg_decoder->CreateNVJpeg(ffmpeg_source->source_desc));
                nvjpeg_decoder->vSource = ffmpeg_source;
                return S_OK;
        }
        return E_INVALIDARG;
    }

    return S_OK;
}

winrt::hresult __stdcall HAV::CreateDevice(REFIID iid, DEV_DESC dev_desc, IDev** Out)
{
    if (IsEqualIID(iid, IID_HAV_NVDev)) {
        winrt::com_ptr<DevNVIDIA> dev_ptr;
        try {
            winrt::check_pointer(Out);
            dev_ptr = winrt::make_self<DevNVIDIA>();
            winrt::check_pointer(dev_ptr.get());
            winrt::check_hresult(dev_ptr->InitDevice(dev_desc));
            *Out = dev_ptr.get();
            dev_ptr.detach();
            return S_OK;
        }
        catch (winrt::hresult_error const& err) {
            return err.code();
        }
    }

    return E_NOINTERFACE;
}

winrt::hresult __stdcall HAV::CreateDemuxer(REFIID iid, IDemuxer** Out)
{
    winrt::com_ptr<IDemuxer> demx_ptr;
    if (IsEqualIID(iid, IID_HAV_FFMPEGDemuxer)) {
        try {
            winrt::check_pointer(Out);
            demx_ptr = winrt::make_self<FFMPEGDemuxer>();
            winrt::check_pointer(demx_ptr.get());
            *Out = demx_ptr.get();
            demx_ptr.detach();
            return S_OK;
        }catch (winrt::hresult_error const& err) {
            return err.code();
        }
    }

    return E_NOINTERFACE;
}

winrt::hresult __stdcall HAV::CreateDisplay(REFIID iid, IDisplay** Out)
{
    if (IsEqualIID(iid, IID_HAV_WinDisplay))
    {
        winrt::com_ptr<WinDisplay> display_ptr;
        try {
            winrt::check_pointer(Out);
            display_ptr = winrt::make_self<WinDisplay>();
            winrt::check_pointer(display_ptr.get());
            display_ptr->ConfigureDisplay();
            *Out = display_ptr.get();
            display_ptr.detach();
            return S_OK;
        } catch (winrt::hresult_error const& err) {
            return err.code();
        }
    }
    return winrt::hresult();
}

winrt::hresult __stdcall HAV::CreateDecoder(REFIID iid, IDecoder** Out)
{
    winrt::com_ptr<IDecoder> dec_ptr;
    if (IsEqualIID(iid, IID_HAV_NVDEC))
    {
        try {
            winrt::check_pointer(Out);
            dec_ptr = winrt::make_self<NVDEC>();
            winrt::check_pointer(dec_ptr.get());
            *Out = dec_ptr.get();
            dec_ptr.detach();
            return S_OK;
        }catch(winrt::hresult_error const& err) {
            return err.code();
        }
    }

    if (IsEqualIID(iid, IID_HAV_NVJpegDecoder))
    {
        try {
            winrt::check_pointer(Out);
            dec_ptr = winrt::make_self<NVJpegDecoder>();
            winrt::check_pointer(dec_ptr.get());
            *Out = dec_ptr.get();
            dec_ptr.detach();
            return S_OK;
        }
        catch (winrt::hresult_error const& err) {
            return err.code();
        }
    }
    return E_NOINTERFACE;
}

winrt::hresult __stdcall HAV::CreateEncoder(REFIID iid, IEncoder** Out)
{
    return S_OK;
}

winrt::hresult __stdcall HAV::CreateFrame(REFIID iid, FRAME_OUTPUT_DESC frame_desc, IFrame** Out)
{
    if (IsEqualIID(iid, IID_HAV_NVFrame))
    {
        winrt::com_ptr<NVFrame> frame_ptr;
        try {
            winrt::check_pointer(Out);
            frame_ptr = winrt::make_self<NVFrame>();
            winrt::check_pointer(frame_ptr.get());
            winrt::check_hresult(frame_ptr->ConfigureFrame(frame_desc));
            frame_ptr->cuDesc = frame_desc;
            *Out = frame_ptr.get();
            frame_ptr.detach();
            return S_OK;
        }
        catch (winrt::hresult_error const& err) {
            return err.code();
        }
    }
}
