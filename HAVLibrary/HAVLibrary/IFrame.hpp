#pragma once
#include "pch.hpp"
#include "HAVTypes.hpp"
#include "IHAVComponent.hpp"

// {78FB1499-1525-4C9F-91F1-6239AE57A897}
DEFINE_GUID(IID_HAV_IFrame,
	0x78fb1499, 0x1525, 0x4c9f, 0x91, 0xf1, 0x62, 0x39, 0xae, 0x57, 0xa8, 0x97);
struct ID3D11Resource;
struct FRAME_OUTPUT_DESC
{
	unsigned int width;
	unsigned int height;
	HVFormat format;
	HVColorSpace colorspace;
	float wr;
	float wb;
	HVTransfer transfer;
	HVToneMapper tone_mapper;
	float max_content_luminance;
	float display_luminance;
};
class __declspec(uuid("78FB1499-1525-4C9F-91F1-6239AE57A897")) IFrame : public IUnknown
{
public:
	virtual winrt::hresult GetDesc(FRAME_OUTPUT_DESC &desc) = 0;
	virtual winrt::hresult ConvertFormat(HVFormat fmt, IFrame *out) = 0;
	//virtual winrt::hresult RegisterD3D9Resource(IDirect3DResource* resource) = 0;
	virtual winrt::hresult RegisterD3D11Resource(ID3D11Resource* resource) = 0;
	virtual winrt::hresult CommitResource() = 0;
};