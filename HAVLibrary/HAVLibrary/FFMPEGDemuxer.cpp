#include "FFMPEGDemuxer.hpp"
#include <iostream>

#pragma comment(lib, "avcodec")
#pragma comment(lib, "avformat")
#pragma comment(lib, "avutil")

winrt::hresult FFMPEGDemuxer::VideoCapture(int ordinal)
{
	return S_OK;
}

winrt::hresult FFMPEGDemuxer::VideoCapture(std::string path, IVideoSource** source)
{
	FVContext new_context = { 0 };
	VIDEO_SOURCE_DESC stream_info;
	int vstream_index = 0;

	winrt::check_hresult(AVHr(avformat_open_input(&new_context.av_fmt_context, path.c_str(), nullptr, nullptr)));
	winrt::check_hresult(AVHr(avformat_find_stream_info(new_context.av_fmt_context, nullptr)));

	vstream_index = av_find_best_stream(new_context.av_fmt_context, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
	if (vstream_index < 0)
		return E_INVALIDARG;

	new_context.streamIndex = vstream_index;
	new_context.vstream = new_context.av_fmt_context->streams[vstream_index];

	stream_info.codec = AVCOHAV(new_context.vstream->codecpar->codec_id);
	stream_info.duration = ((double)new_context.vstream->duration * ((double)new_context.vstream->time_base.num) / (double)new_context.vstream->time_base.den);
	stream_info.framerate = ((double)new_context.vstream->avg_frame_rate.num /
		(double)new_context.vstream->avg_frame_rate.den);
	stream_info.width = new_context.vstream->codecpar->width;
	stream_info.heigth = new_context.vstream->codecpar->height;
	stream_info.chroma = AVCHAV(new_context.vstream->codecpar->format);
	stream_info.bitdepth = new_context.vstream->codecpar->bits_per_raw_sample;
	

	if (stream_info.codec == HV_CODEC_H264 || stream_info.codec == HV_CODEC_H265_420) {
		if (stream_info.codec == HV_CODEC_H264)
			new_context.av_bsf_filter = av_bsf_get_by_name("h264_mp4toannexb");
		if (stream_info.codec == HV_CODEC_H265_420)
			new_context.av_bsf_filter = av_bsf_get_by_name("hevc_mp4toannexb");

		winrt::check_hresult(AVHr(av_bsf_alloc(new_context.av_bsf_filter, &new_context.av_bsf_context)));
		new_context.av_bsf_context->par_in = new_context.vstream->codecpar;
		winrt::check_hresult(AVHr(av_bsf_init(new_context.av_bsf_context)));
	}

	winrt::com_ptr<FFMPEGVideoSource> private_source = winrt::make_self<FFMPEGVideoSource>();
	winrt::check_pointer(private_source.get());
	private_source->InitSource(new_context, stream_info);
	*source = private_source.get();
	private_source.detach();

	return S_OK;
}

winrt::hresult FFMPEGDemuxer::VideoCapture(std::string IP, unsigned short port)
{
	return winrt::hresult();
}

winrt::hresult FFMPEGDemuxer::DesktopCapture(int monitor)
{
	return E_NOTIMPL;
}