#include "FFMPEGCaptureSource.hpp"

winrt::hresult FFMPEGCaptureSource::GetDesc(VIDEO_SOURCE_DESC& desc)
{
	desc = source_desc;
	return S_OK;
}

void FFMPEGCaptureSource::InitSource(FVContext ctx, VIDEO_SOURCE_DESC source)
{
	source_ctx.av_fmt_context = ctx.av_fmt_context;
	source_ctx.av_inp_format = ctx.av_inp_format;
	source_ctx.av_bsf_context = ctx.av_bsf_context;
	source_ctx.vstream = ctx.vstream;

	source_desc.codec = source.codec;
	source_desc.duration = source.duration;
	source_desc.framerate = source.framerate;
	source_desc.width = source.width;
	source_desc.heigth = source.heigth;
}
