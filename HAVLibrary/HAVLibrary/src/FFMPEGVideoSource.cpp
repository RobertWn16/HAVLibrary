#include "FFMPEGVideoSource.hpp"
#include <iostream>
FFMPEGVideoSource::~FFMPEGVideoSource()
{
	if (av_pck.data)
		av_packet_unref(&av_pck);
	if (flt_pck.data)
		av_packet_unref(&flt_pck);
	if (source_ctx.av_fmt_context)
		avformat_close_input(&source_ctx.av_fmt_context);

}
winrt::hresult FFMPEGVideoSource::GetDesc(VIDEO_SOURCE_DESC& desc)
{
	desc = source_desc;
	return S_OK;
}

winrt::hresult FFMPEGVideoSource::Parse(IPacket* packet)
{
	FFMPEGPacket* ffmpeg_packet = dynamic_cast<FFMPEGPacket*>(packet);
	winrt::check_pointer(ffmpeg_packet);
	while (true)
	{
		if (av_pck.data)
			av_packet_unref(&av_pck);
		if (SUCCEEDED(AVHr(av_read_frame(source_ctx.av_fmt_context, &av_pck)))) {
			if (av_pck.stream_index == source_ctx.streamIndex) {
				if (source_desc.codec == HV_CODEC_H264 || source_desc.codec == HV_CODEC_H265_420 ||
					source_desc.codec == HV_CODEC_VP9) {

					if (flt_pck.data)
						av_packet_unref(&flt_pck);
					winrt::check_hresult(AVHr(av_bsf_send_packet(source_ctx.av_bsf_context, &av_pck)));
					winrt::check_hresult(AVHr(av_bsf_receive_packet(source_ctx.av_bsf_context, &flt_pck)));
					winrt::check_pointer(flt_pck.data);

					ffmpeg_packet->RecievePacket(&flt_pck);
					return S_OK;
				}
				ffmpeg_packet->RecievePacket(&av_pck);
				return S_OK;
			}
		}
	}
}

winrt::hresult FFMPEGVideoSource::Parse(ID3D11Texture2D** Out) noexcept
{
	return E_NOINTERFACE;
}

void FFMPEGVideoSource::InitSource(FVContext ctx, VIDEO_SOURCE_DESC source)
{
	source_ctx.av_fmt_context = ctx.av_fmt_context;
	source_ctx.av_bsf_context = ctx.av_bsf_context;
	source_ctx.vstream = ctx.vstream;

	source_desc = source;
}
