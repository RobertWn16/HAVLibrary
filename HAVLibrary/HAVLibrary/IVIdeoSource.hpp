#pragma once
#include "IHAVComponent.hpp"
#include "HAVTypes.hpp"

// {3E0E196D-A155-431F-AFA1-1BDA94298955}
DEFINE_GUID(IID_HAV_IVCaptureSource,
	0x3e0e196d, 0xa155, 0x431f, 0xaf, 0xa1, 0x1b, 0xda, 0x94, 0x29, 0x89, 0x55);

struct VIDEO_SOURCE_DESC
{
	double duration;
	double framerate;
	HVCodec codec;
	HVChroma chroma;
	HVFormat format;
	unsigned int bitdepth;
	unsigned int width;
	unsigned int heigth;
};

class __declspec(uuid("3E0E196D-A155-431F-AFA1-1BDA94298955")) IVideoSource : public IHAVComponent
{
public:
	virtual winrt::hresult GetDesc(VIDEO_SOURCE_DESC &desc) = 0;
	virtual winrt::hresult Parse(void *desc) = 0;
};