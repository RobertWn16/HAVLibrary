#include "HAV.hpp"

winrt::hresult HAV::Link(IHAVComponent* In, IHAVComponent* Out)
{
    DevNVIDIA* dev_nvidia = dynamic_cast<DevNVIDIA*>(In);
    if (dev_nvidia) {
        NVDEC* nv_decoder = dynamic_cast<NVDEC*>(Out);
        if (nv_decoder) {
            if (!nv_decoder->hasDevice) {
                nv_decoder->deviceContext = dev_nvidia->cuContext;
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
        return E_INVALIDARG;
    }

    return S_OK;
}
