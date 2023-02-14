#include "FFMPEGMuxer.hpp"

FFMPEGMuxer::~FFMPEGMuxer()
{
    av_write_trailer(oc);
    avio_close(oc->pb);
    //avformat_free_context(oc);
}

winrt::hresult FFMPEGMuxer::VideoStream()
{
    int err;
    oc = avformat_alloc_context();
    const AVOutputFormat* fmt = av_guess_format("h264", NULL, NULL);
    oc->oformat = fmt;
    oc->url = (char*)"sample.h264";


    vs = avformat_new_stream(oc, NULL);

    vs->id = 0;

    // Set video parameters
    AVCodecParameters* vpar = vs->codecpar;
    vpar->codec_id = AV_CODEC_ID_H264;
    vpar->codec_type = AVMEDIA_TYPE_VIDEO;
    vpar->width = 3840;
    vpar->height = 2160;

    err = avio_open(&oc->pb, oc->url, AVIO_FLAG_WRITE);
    err = avformat_write_header(oc, NULL);

    pkt = av_packet_alloc();

    return S_OK;
}

winrt::hresult FFMPEGMuxer::Stream(uint8_t* data, unsigned int size, int fps)
{
    pkt->dts = pkt->pts;
    pkt->stream_index = vs->index;
    pkt->data = data;
    pkt->size = size;

    av_write_frame(oc, pkt);
    av_write_frame(oc, NULL);
    return S_OK;
}
