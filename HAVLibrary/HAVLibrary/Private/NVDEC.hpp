#pragma once
#include "devNVIDIA.hpp"
#include "IDecoder.hpp"
#include "HAVUtilsPrivate.hpp"
#include "NVFrame.hpp"
#include "IPacket.hpp"

struct NVDEC : winrt::implements<NVDEC, IDecoder>
{
private:
	VIDEO_SOURCE_DESC cuVideo_desc;
	CUvideodecoder cuDecoder;
	CUvideoparser cuParser;
	cudaVideoCodec cuvidCodec;
	CUcontext deviceContext;
	std::mutex cuLock;
	std::queue<CUdeviceptr> cuBuffer;
	CUdeviceptr dec_bkbuffer = 0;
	unsigned int surfaceHeigth = 0;

	static int parser_decode_picture_callback(void* pUser, CUVIDPICPARAMS* pic);
	static int parser_sequence_callback(void* pUser, CUVIDEOFORMAT* fmt);
	static int parser_display_picture_callback(void* pUser, CUVIDPARSERDISPINFO* info) noexcept;
	static int parser_get_operation_point_callback(void* pUser, CUVIDOPERATINGPOINTINFO* opInfo);

public:
	~NVDEC();
	winrt::hresult IsSupported(VIDEO_SOURCE_DESC desc) final;
	winrt::hresult Decode(IPacket* in, IFrame *out) final;
	winrt::hresult Decode(unsigned char* buf, unsigned int length, unsigned int timestamp, IFrame* out) final;
	winrt::hresult CreateParser(VIDEO_SOURCE_DESC desc);

public:
	winrt::hresult ConfigureDecoder(VIDEO_SOURCE_DESC vsrc_desc, IDev *dev) final;

public:
	IVideoSource* vSource;
	bool hasDevice = false;
	bool hasSource = false;
};

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
	return 20;
}