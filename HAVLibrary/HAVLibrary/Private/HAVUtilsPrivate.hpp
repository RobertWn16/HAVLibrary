#pragma once
#include "pch.hpp"
#include "HAVTypes.hpp"
extern "C"
{
	#include <libavformat/avformat.h>
	#include <libavcodec/avcodec.h>
	#include <libavutil/avutil.h>
	#include <libavutil/mastering_display_metadata.h>
	#include <libavcodec/bsf.h>
}
#include <nvcuvid.h>
#include <cuda.h>
#include <d3d11.h>
#pragma comment(lib, "nvcuvid")
#pragma comment(lib, "cuda")
#pragma comment(lib, "d3d11")


winrt::hresult AVHr(int avcode);

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
	case AV_CODEC_ID_MJPEG:
		return HV_CODEC_MJPEG;
	default:
		break;
	}

	return HV_CODEC_UNSUPPORTED;
}

HVFormat DXGIFmtHV(DXGI_FORMAT dxgi_format);
HVColorSpace DXGICsHV(DXGI_COLOR_SPACE_TYPE dxgi_colorspace);
/*static void HAV_LOG(std::string componentName, winrt::hresult code, winrt::hstring errName)
{
	std::cout << "[" << componentName << " ]: " << errName << " -> Error code 0x%x";
}*/