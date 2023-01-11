#pragma once
#include "pch.hpp"
#include "HAVTypes.hpp"

extern "C"
{
	#include <libavformat/avformat.h>
	#include <libavcodec/avcodec.h>
	#include <libavutil/avutil.h>
	#include <libavcodec/bsf.h>
}

static winrt::hresult AVHr(int avcode)
{
	char buf[BUFSIZ];
	av_strerror(avcode, buf, AV_ERROR_MAX_STRING_SIZE);
	if (avcode < 0)
		return E_FAIL;
	return S_OK;
}

static HVCodec AVCOHAV(int avcodec)
{
	switch (avcodec)
	{
	case AV_CODEC_ID_MPEG1VIDEO:
		return HV_CODEC_MPEG1;
	case AV_CODEC_ID_MPEG2VIDEO:
		return HV_CODEC_MPEG2;
	case AV_CODEC_ID_VC1:
		return HV_CODEC_VC1;
	case AV_CODEC_ID_VP8:
		return HV_CODEC_VP8;
	case AV_CODEC_ID_VP9:
		return HV_CODEC_VP9;
	case AV_CODEC_ID_H264:
		return HV_CODEC_H264;
	case AV_CODEC_ID_HEVC:
		return HV_CODEC_H265_420;
	case AV_CODEC_ID_AV1:
		return HV_CODEC_AV1;
	default:
		break;
	}

	return HV_CODEC_UNSUPPORTED;
}

struct PACKET_DESC
{
	void* data;
	int32_t size;
};