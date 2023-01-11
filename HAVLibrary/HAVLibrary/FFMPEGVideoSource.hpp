#pragma once
#include "IVIdeoSource.hpp"
#include "IDecoder.hpp"
#include "HAVUtilsPrivate.hpp"

struct FVContext
{
public:
	AVFormatContext* av_fmt_context = nullptr;
	AVInputFormat* av_inp_format = nullptr;
	AVBSFContext* av_bsf_context = nullptr;
	const AVBitStreamFilter* av_bsf_filter = nullptr;
	unsigned int streamIndex = 0;
	AVStream* vstream = nullptr;
};

struct FFMPEGVideoSource : winrt::implements<FFMPEGVideoSource, IVideoSource>
{
public:
	IDecoder* source_decoder;
	AVPacket av_pck;
	AVPacket flt_pck;
	FVContext source_ctx;
	VIDEO_SOURCE_DESC source_desc;
	bool sync;

public:
	winrt::hresult GetDesc(VIDEO_SOURCE_DESC &desc) final;
	winrt::hresult Parse(void* desc) final;

public:
	void InitSource(FVContext ctx, VIDEO_SOURCE_DESC desc);
};

static HVChroma AVCHAV(int avchroma)
{
	switch (avchroma)
	{
	case AV_PIX_FMT_YUV420P:
		return HV_CHROMA_FORMAT_420;
	case AV_PIX_FMT_YUV422P:
		return HV_CHROMA_FORMAT_422;
	case AV_PIX_FMT_YUV444P:
		return HV_CHROMA_FORMAT_444;
	default:
		break;
	}
}

