#pragma once
#include "IEncoder.hpp"
#include "HAVUtilsPrivate.hpp"
#include "NVFrame.hpp"
#include "FFMPEGPacket.hpp"
#include <nvEncodeAPI.h>

struct NVENCQuality
{
	GUID nvenc_presetGuid;
	NV_ENC_TUNING_INFO nvenc_tuningInfo;
};

struct NVENC : winrt::implements<NVENC, IEncoder>
{
private:
	NV_ENCODE_API_FUNCTION_LIST nvenc = { NV_ENCODE_API_FUNCTION_LIST_VER };
	NV_ENC_REGISTERED_PTR nvencRegisteredPtr;
	NV_ENC_OUTPUT_PTR nvencBistreamOutPtr;
	HANDLE nvencCompletedFrame;
	CUdeviceptr nvencFrame = NULL;
	void* nvencPtr = nullptr;
	CUcontext deviceCtx;
	size_t nvencFramePitch = 0;
	ENCODER_DESC nvencDesc;

public:
	~NVENC();
	winrt::hresult IsSupported(VIDEO_SOURCE_DESC desc);
	winrt::hresult Encode(IFrame* inFrame);
	winrt::hresult GetEncodedPacket(IPacket* outPacket);
	winrt::hresult ConfigureEncoder(ENCODER_DESC nvencDesc, CUcontext deviceContext);
};
