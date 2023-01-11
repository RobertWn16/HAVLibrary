#pragma once
#include "IDecoder.hpp"
#include "HAVUtilsPrivate.hpp"
#include <cuda.h>
#include <nvcuvid.h>

#pragma comment(lib, "nvcuvid")
#pragma comment(lib, "cuda")

struct NVDEC : winrt::implements<NVDEC, IDecoder>
{
private:
	CUvideodecoder cuDecoder;
	CUvideoparser cuParser;
	cudaVideoCodec cuvidCodec;
	std::mutex cuLock;
	std::queue<CUdeviceptr> cuBuffer;

public:
	winrt::hresult IsSupported(VIDEO_SOURCE_DESC desc) final;
	winrt::hresult Decode(IFrame **out) final;

	winrt::hresult CreateParser(VIDEO_SOURCE_DESC desc);

private:
	static int parser_decode_picture_callback(void* pUser, CUVIDPICPARAMS* pic);
	static int parser_sequence_callback(void* pUser, CUVIDEOFORMAT* fmt);
	static int parser_display_picture_callback(void* pUser, CUVIDPARSERDISPINFO* info);

public:
	IVideoSource* vSource;
	CUcontext deviceContext;
	bool hasDevice = false;
	bool hasSource = false;
};

//Utils
static cudaVideoCodec HAVCONV(int havcodec)
{
	switch (havcodec)
	{
	case HV_CODEC_MPEG1:
		return cudaVideoCodec_MPEG1;
	case HV_CODEC_MPEG2:
		return cudaVideoCodec_MPEG2;
	case HV_CODEC_VC1:
		return cudaVideoCodec_VC1;
	case HV_CODEC_VP8:
		return cudaVideoCodec_VP8;
	case HV_CODEC_VP9:
		return cudaVideoCodec_VP9;
	case HV_CODEC_H264:
		return cudaVideoCodec_H264;
	case HV_CODEC_H265_420:
		return cudaVideoCodec_HEVC;
	case HV_CODEC_AV1:
		return cudaVideoCodec_AV1;
	default:
		break;
	}
}

static cudaVideoChromaFormat HAVCHNV(int havchroma)
{
	switch (havchroma)
	{
	case HV_CHROMA_FORMAT_420:
		return cudaVideoChromaFormat_420;
	case HV_CHROMA_FORMAT_422:
		return cudaVideoChromaFormat_422;
	case HV_CHROMA_FORMAT_444:
		return cudaVideoChromaFormat_444;
	}
}

static unsigned long GetNumDecodeSurfaces(cudaVideoCodec eCodec, unsigned int nWidth, unsigned int nHeight)
{
	if (eCodec == cudaVideoCodec_VP9) {
		return 12;
	}

	if (eCodec == cudaVideoCodec_H264 || eCodec == cudaVideoCodec_H264_SVC || eCodec == cudaVideoCodec_H264_MVC) {
		// assume worst-case of 20 decode surfaces for H264
		return 20;
	}

	if (eCodec == cudaVideoCodec_HEVC) {
		// ref HEVC spec: A.4.1 General tier and level limits
		// currently assuming level 6.2, 8Kx4K
		int MaxLumaPS = 35651584;
		int MaxDpbPicBuf = 6;
		int PicSizeInSamplesY = (int)(nWidth * nHeight);
		int MaxDpbSize;
		if (PicSizeInSamplesY <= (MaxLumaPS >> 2))
			MaxDpbSize = MaxDpbPicBuf * 4;
		else if (PicSizeInSamplesY <= (MaxLumaPS >> 1))
			MaxDpbSize = MaxDpbPicBuf * 2;
		else if (PicSizeInSamplesY <= ((3 * MaxLumaPS) >> 2))
			MaxDpbSize = (MaxDpbPicBuf * 4) / 3;
		else
			MaxDpbSize = MaxDpbPicBuf;
		return MaxDpbSize + 4;
	}
	return 8;
}