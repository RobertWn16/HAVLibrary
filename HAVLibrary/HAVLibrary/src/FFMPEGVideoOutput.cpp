#include "FFMPEGVideoOutput.hpp"

FFMPEGVideoOutput::~FFMPEGVideoOutput()
{
    av_write_trailer(ffmpegOutputContext);
    //avformat_free_context(ffmpegOutputContext);
}

winrt::hresult FFMPEGVideoOutput::ConfigureVideoOutput(AVFormatContext* oc, VIDEO_OUTPUT_DESC outDesc)
{
    try
    {
        winrt::check_pointer(oc);
        ffmpegOutputContext = oc;
        ffmpegOutputDesc = outDesc;
        ffmpegOutputStream = avformat_new_stream(oc, nullptr);

        oc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        oc->video_codec_id = AV_CODEC_ID_H264;

        ffmpegOutputStream->id = 0;
        ffmpegOutputStream->codecpar->codec_id = AV_CODEC_ID_H264;
        ffmpegOutputStream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
        ffmpegOutputStream->codecpar->width = ffmpegOutputDesc.width;
        ffmpegOutputStream->codecpar->height = ffmpegOutputDesc.height;
        
        ffmpegOutputStream->time_base.num = 1;
        ffmpegOutputStream->time_base.den = 30;
        ffmpegFps = ffmpegOutputStream->time_base.den;

        winrt::check_hresult(AVHr(avio_open(&oc->pb, oc->url, AVIO_FLAG_WRITE)));
        winrt::check_hresult(AVHr(avformat_write_header(oc, nullptr)));

    } catch (winrt::hresult_error const& err) {
        return err.code();
    }
    return S_OK;
}

winrt::hresult FFMPEGVideoOutput::Write(IPacket* inPck)
{
    try
    {
        FFMPEGPacket* ffmpeg_packet = dynamic_cast<FFMPEGPacket*>(inPck);
        winrt::check_pointer(ffmpeg_packet);
        AVPacket* ffmpeg_internal_packet = ffmpeg_packet->GetPacket();
        ffmpeg_internal_packet->pts = av_rescale_q(ffmpegPts++, { 1, ffmpegFps }, ffmpegOutputStream->time_base);
        ffmpeg_internal_packet->dts = ffmpeg_internal_packet->pts;
        ffmpeg_internal_packet->stream_index = 0;
        av_write_frame(ffmpegOutputContext, ffmpeg_internal_packet);
        av_write_frame(ffmpegOutputContext, nullptr);

    } catch (winrt::hresult_error const& err) {
        return err.code();
    }

    return winrt::hresult();
}
