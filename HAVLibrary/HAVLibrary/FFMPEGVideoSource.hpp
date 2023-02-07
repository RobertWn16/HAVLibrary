#pragma once
#include "IVIdeoSource.hpp"
#include "IDecoder.hpp"
#include "HAVUtilsPrivate.hpp"

struct FVContext
{
public:
	AVFormatContext* av_fmt_context = nullptr;
	AVBSFContext* av_bsf_context = nullptr;
	const AVBitStreamFilter* av_bsf_filter = nullptr;
	unsigned int streamIndex = 0;
	AVStream* vstream = nullptr;
};

struct FFMPEGVideoSource : winrt::implements<FFMPEGVideoSource, IVideoSource>
{
private:
	IDecoder* source_decoder;
	AVPacket av_pck;
	AVPacket flt_pck;
	FVContext source_ctx;
	bool sync;

public:
	VIDEO_SOURCE_DESC source_desc;

	~FFMPEGVideoSource();

	winrt::hresult GetDesc(VIDEO_SOURCE_DESC &desc) final;
	winrt::hresult Parse(void* desc) final;
	void InitSource(FVContext ctx, VIDEO_SOURCE_DESC desc);
};

static HVColorSpace AVcsHAVcs(AVColorPrimaries av_primaries)
{
	switch (av_primaries)
	{
	case AVCOL_PRI_RESERVED0:

		break;
	case AVCOL_PRI_BT709:
		return HV_COLORSPACE_BT709;
		break;
	case AVCOL_PRI_BT2020:
		return HV_COLORSPACE_BT2020;
		break;
	default:
		break;
	}

	return HV_COLORSPACE_UNKNOWN;
}
static HVTransfer AVTrHAVTransfer(AVColorTransferCharacteristic av_transfer)
{
	switch (av_transfer)
	{
	case AVCOL_TRC_UNSPECIFIED:
		return HV_TRANSFER_UNKNOWN;
		break;
	case AVCOL_TRC_SMPTE2084:
		return HV_TRANSFER_PQ;
		break;
	case AVCOL_TRC_ARIB_STD_B67:
		return HV_TRANSFER_HLG;
		break;
	default:
		break;
	}
}
static HVChroma AVFmtChHAV(int avchroma, unsigned int &bitdepth)
{
	HVChroma hvChroma = HV_CHROMA_FORMAT_UNKNOWN;

	switch (avchroma)
	{
	case AV_PIX_FMT_YUV420P10LE:
	case AV_PIX_FMT_YUV420P12LE:
	case AV_PIX_FMT_YUV420P:
		hvChroma = HV_CHROMA_FORMAT_420;
		break;

	case AV_PIX_FMT_YUV422P10LE:
	case AV_PIX_FMT_YUV422P12LE:
	case AV_PIX_FMT_YUV422P:
		hvChroma = HV_CHROMA_FORMAT_422;
		break;

	case AV_PIX_FMT_YUV444P10LE:
	case AV_PIX_FMT_YUV444P12LE:
	case AV_PIX_FMT_YUV444P:
		hvChroma = HV_CHROMA_FORMAT_444;
		break;
	default:
		break;
	}

	switch (avchroma)
	{
	case AV_PIX_FMT_YUV420P:
	case AV_PIX_FMT_YUV422P:
	case AV_PIX_FMT_YUV444P:
		bitdepth = 8;
		break;

	case AV_PIX_FMT_YUV420P10LE:
	case AV_PIX_FMT_YUV422P10LE:
	case AV_PIX_FMT_YUV444P10LE:
		bitdepth = 10;
		break;

	case AV_PIX_FMT_YUV420P12LE:
	case AV_PIX_FMT_YUV422P12LE:
	case AV_PIX_FMT_YUV444P12LE:
		bitdepth = 12;
		break;

	default:
		break;
	}

	return hvChroma;
}

