#pragma once
#include "IEncoder.hpp"
#include "HAVUtilsPrivate.hpp"
#include "NVFrame.hpp"
#include "FFMPEGPacket.hpp"
#include <nvEncodeAPI.h>

constexpr int NVEC_INTERNAL_EVENTS_NUM = 2;
constexpr int NVENC_SAFE_GRAB = 0;
constexpr int NVENC_SESSION_CLOSED = 1;

struct NVENCQuality
{
	GUID nvenc_presetGuid;
	NV_ENC_TUNING_INFO nvenc_tuningInfo;
};

struct NVENC : winrt::implements<NVENC, IEncoder>
{
private:
	NV_ENCODE_API_FUNCTION_LIST nvenc = { NV_ENCODE_API_FUNCTION_LIST_VER };
	NV_ENC_REGISTERED_PTR* nvencRegisteredPtr;
	NV_ENC_OUTPUT_PTR* nvencBistreamOutPtr;
	HANDLE nvencInternalEvents[NVEC_INTERNAL_EVENTS_NUM];
	HANDLE* nvencCompletedFrame;

	CUdeviceptr* nvencFrame = NULL;
	NV_ENC_INPUT_PTR* nvencMappedResource = NULL;
	void* nvencPtr = nullptr;
	CUcontext nvencCtx;
	size_t nvencFramePitch = 0;
	ENCODER_DESC nvencDesc;
	unsigned int nvencTxInIndex = 0;
	unsigned int nvencTxOutIndex = 0;
	unsigned int nvencRxIndex = 0;

	std::queue<NV_ENC_OUTPUT_PTR> nvencBitstreamRefQueue;
	std::queue<HANDLE> nvencCompletionRefQueue;

public:
	~NVENC();
	winrt::hresult IsSupported(VIDEO_SOURCE_DESC desc);
	winrt::hresult Encode(IFrame* inFrame);
	winrt::hresult GetEncodedPacket(IPacket* outPacket);
	winrt::hresult GetSequenceParams(unsigned int* size, void** spps);
	winrt::hresult ConfigureEncoder(ENCODER_DESC nvencDesc, CUcontext deviceContext);
};
