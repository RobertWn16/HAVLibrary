#pragma once
#include "IEncoder.hpp"
#include "HAVUtilsPrivate.hpp"
#include "NVFrame.hpp"
#include <nvEncodeAPI.h>

struct NVENC : winrt::implements<NVENC, IEncoder>
{
private:
	NV_ENCODE_API_FUNCTION_LIST functionList = { NV_ENCODE_API_FUNCTION_LIST_VER };
	NV_ENC_INITIALIZE_PARAMS nvencInitPar;
	NV_ENC_CREATE_INPUT_BUFFER nv_enc_input_buffer;
	NV_ENC_CREATE_BITSTREAM_BUFFER nv_enc_bitstream;
	CUdeviceptr frame = NULL;
	void *nvencEncoder = nullptr;
	NV_ENC_REGISTERED_PTR nv_enc_res;
	NV_ENC_INPUT_PTR nv_enc_input;
	CUcontext deviceCtx;
	size_t pitch = 0;
	FILE* output_test = nullptr;

public:
	~NVENC();
	winrt::hresult IsSupported(VIDEO_SOURCE_DESC desc);
	winrt::hresult Encode(IFrame* in, unsigned char* buf, unsigned int& size);
	winrt::hresult ConfigureEncoder(CUcontext deviceContext);
};