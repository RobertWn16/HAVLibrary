#include "FFMPEGMuxer.hpp"

FFMPEGMuxer::~FFMPEGMuxer()
{
    av_write_trailer(oc);
    avformat_free_context(oc);
}

winrt::hresult FFMPEGMuxer::VideoStream()
{
    int err = 0;
    AVDictionary* options = NULL;

    const AVOutputFormat* fmt = nullptr;
    AVCodecParameters* c;

    char* filename = (char*)"tcp://127.0.0.1:8080/live.h264?listen";
    avformat_network_init();

    err = avformat_alloc_output_context2(&oc, NULL, "h264", filename);
    st = avformat_new_stream(oc, nullptr);

    st->id = 0;
    st->codecpar->codec_id = AV_CODEC_ID_H264;
    st->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    st->codecpar->width = 3840;
    st->codecpar->height = 2160;
    st->time_base.num = 1;
    st->time_base.den = 60;
    err = av_dict_set(&options, "codec", "copy", 0);
    err = avio_open2(&oc->pb, filename, AVIO_FLAG_WRITE, nullptr, &options);
    oc->video_codec_id = AV_CODEC_ID_H264;

    oc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    err = avformat_write_header(oc, nullptr);


    av_dump_format(oc, 0, "filename", 1);
    return S_OK;
}

winrt::hresult FFMPEGMuxer::Stream(IPacket* inPkt)
{
    static int nPts = 0;
    int err = 0;
    FFMPEGPacket* ffmpeg_pkt = dynamic_cast<FFMPEGPacket*>(inPkt);
    ffmpeg_pkt->pkt->pts = av_rescale_q(nPts++, AVRational{ 1, 60 }, st->time_base);
    ffmpeg_pkt->pkt->dts = ffmpeg_pkt->pkt->pts;
    ffmpeg_pkt->pkt->stream_index = 0;
    err = av_write_frame(oc, ffmpeg_pkt->pkt);
    err = av_write_frame(oc, nullptr);
    return S_OK;
}
