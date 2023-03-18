#pragma once
#include "IDev.hpp"
#include "IVIdeoSource.hpp"
#include "IFrame.hpp"
// {E4EA258B-86C8-4528-9AFE-C1DCE01ACB1A}
DEFINE_GUID(IID_HAV_IDecoder,
	0xe4ea258b, 0x86c8, 0x4528, 0x9a, 0xfe, 0xc1, 0xdc, 0xe0, 0x1a, 0xcb, 0x1a);

struct DECODER_DESC
{
	unsigned int width;
	unsigned int heigth;
	HVCodec codec;
	unsigned int bitdepth;
};
class __declspec(uuid("E4EA258B-86C8-4528-9AFE-C1DCE01ACB1A")) IDecoder : public IHAVComponent
{
public:
	virtual winrt::hresult IsSupported(VIDEO_SOURCE_DESC desc) = 0;
	virtual winrt::hresult Decode(IPacket* in, IFrame *out) = 0;
	virtual winrt::hresult Decode(unsigned char* buf, unsigned int length, unsigned int timestamp, IFrame* out) = 0;
protected:
	virtual winrt::hresult ConfigureDecoder(VIDEO_SOURCE_DESC desc, IDev* dev) = 0;
};