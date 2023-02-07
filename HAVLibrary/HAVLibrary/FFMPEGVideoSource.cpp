#include "FFMPEGVideoSource.hpp"
#include <iostream>
winrt::hresult FFMPEGVideoSource::GetDesc(VIDEO_SOURCE_DESC& desc)
{
	desc = source_desc;
	return S_OK;
}

winrt::hresult FFMPEGVideoSource::Parse(void* desc)
{
	PACKET_DESC* pck_desc = (PACKET_DESC*)desc;
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

					pck_desc->data = flt_pck.data;
					pck_desc->size = flt_pck.size;
					pck_desc->timestamp = flt_pck.pts;

					return S_OK;
				}
				pck_desc->data = av_pck.data;
				pck_desc->size = av_pck.size;
				pck_desc->timestamp = av_pck.pts;
				return S_OK;
			}
		}
	}
}

void FFMPEGVideoSource::InitSource(FVContext ctx, VIDEO_SOURCE_DESC source)
{
	source_ctx.av_fmt_context = ctx.av_fmt_context;
	source_ctx.av_inp_format = ctx.av_inp_format;
	source_ctx.av_bsf_context = ctx.av_bsf_context;
	source_ctx.vstream = ctx.vstream;

	source_desc = source;
}
