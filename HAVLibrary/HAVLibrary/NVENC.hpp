#pragma once
#include "IEncoder.hpp"
#include "HAVUtilsPrivate.hpp"
#include "NVFrame.hpp"
#include <nvEncodeAPI.h>

struct NVENC : winrt::implements<NVENC, IEncoder>
{
private:
	NV_ENC_INITIALIZE_PARAMS nvencInitPar;
	void *nvencEncoder;
public:
	winrt::hresult IsSupported(VIDEO_SOURCE_DESC desc);
	winrt::hresult Encode(IFrame* out);
	winrt::hresult ConfigureEncoder();
};