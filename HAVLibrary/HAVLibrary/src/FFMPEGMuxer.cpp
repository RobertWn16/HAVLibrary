#include "FFMPEGMuxer.hpp"

FFMPEGMuxer::~FFMPEGMuxer()
{
}

static const char* HVConAVCon(HVContainer hvContainer)
{
    switch (hvContainer)
    {
    case HV_CONTAINER_MP4:
        return "mp4";
        break;
    case HV_CONTAINER_MKV:
        return "mkv";
        break;
    case HV_CONTAINER_WEBM:
        return "webm";
        break;
    case HV_CONTAINER_MPEGTS:
        return "mpegts";
        break;
    case HV_CONTAINER_H264:
        return "h264";
        break;
    case HV_CONTAINER_HEVC:
        return "hevc";
        break;
    default:
        break;
    }

    return "invalid";
}

winrt::hresult FFMPEGMuxer::VideoStream(std::string path, VIDEO_OUTPUT_DESC outDesc, IVideoOutput** out)
{
    try
    {
        AVFormatContext* oc = nullptr;
        AVOutputFormat* fmt = nullptr;
        avformat_alloc_output_context2(&oc, nullptr, HVConAVCon(outDesc.container), path.c_str());
        oc->url = path.data();
        winrt::com_ptr<FFMPEGVideoOutput> video_out = winrt::make_self<FFMPEGVideoOutput>();
        winrt::check_pointer(video_out.get());
        video_out->ConfigureVideoOutput(oc, outDesc);
        *out = video_out.detach();
    } catch (winrt::hresult_error const& err)
    {
        return err.code();
    }
    
    return S_OK;
}