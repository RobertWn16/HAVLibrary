#pragma once
#include "IVIdeoSource.hpp"
// {28B716FF-5AB3-4FD3-AE51-032713F1CAE6}
DEFINE_GUID(IID_HAV_IDisplay,
	0x28b716ff, 0x5ab3, 0x4fd3, 0xae, 0x51, 0x3, 0x27, 0x13, 0xf1, 0xca, 0xe6);

struct DISPLAY_DESC
{
	unsigned int width;
	unsigned int heigth;
	unsigned int bitdepth;
	HVColorSpace colorspace;
	HVColorimetry colorimetry;

	//HDR metadata
	float max_display_luminance;
	float avg_display_luminance;
};
class __declspec(uuid("28B716FF-5AB3-4FD3-AE51-032713F1CAE6")) IDisplay : public IHAVComponent
{
public:
	virtual winrt::hresult DisplayCapture(IVideoSource** out) = 0;
	virtual winrt::hresult GetDesc(DISPLAY_DESC &desc) = 0;
};