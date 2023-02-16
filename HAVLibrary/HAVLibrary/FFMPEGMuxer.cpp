#include "FFMPEGMuxer.hpp"

FFMPEGMuxer::~FFMPEGMuxer()
{
    fclose(out);
    //avformat_free_context(oc);
}

winrt::hresult FFMPEGMuxer::VideoStream()
{
    int err = 0;
    AVDictionary* options = NULL;

    AVOutputFormat* fmt = nullptr;
    AVCodecParameters* c;
    AVStream* st;
    char* filename = (char*)"sample.mp4";
    avformat_network_init();

    err = avformat_alloc_output_context2(&oc, fmt, nullptr, filename);

    st = avformat_new_stream(oc, nullptr);

    st->id = 0;
    st->codecpar->codec_id = AV_CODEC_ID_H264;
    st->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    st->codecpar->width = 3840;
    st->codecpar->height = 2160;
    st->time_base.num = 1;
    st->time_base.den = 60;

    //err = avio_open2(&oc->pb, filename, AVIO_FLAG_WRITE, nullptr, &options);
    oc->video_codec_id = AV_CODEC_ID_H264;
    
    oc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    err = avio_open(&oc->pb, filename, AVIO_FLAG_WRITE);
    err = avformat_write_header(oc, nullptr);

    av_dump_format(oc, 0, "sample.mp4", 1);
    return S_OK;
}

winrt::hresult FFMPEGMuxer::Stream(IPacket* inPkt)
{
    int err = 0;
    FFMPEGPacket* ffmpeg_pkt = dynamic_cast<FFMPEGPacket*>(inPkt);

    err = av_write_frame(oc, ffmpeg_pkt->pkt);
    //err = av_write_frame(oc, nullptr);
    return S_OK;
}
