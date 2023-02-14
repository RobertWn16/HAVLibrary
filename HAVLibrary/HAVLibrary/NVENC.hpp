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
	void *nvencEncoder = nullptr;
public:
	winrt::hresult IsSupported(VIDEO_SOURCE_DESC desc);
	winrt::hresult Encode(IFrame* in, IFrame* out);
	winrt::hresult ConfigureEncoder(CUcontext deviceContext);
};