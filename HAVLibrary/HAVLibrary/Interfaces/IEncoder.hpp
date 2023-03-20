#pragma once
#include "IVIdeoSource.hpp"
#include "IFrame.hpp"
#include "IPacket.hpp"

// {F2A728BD-E4D3-4FDD-A6DE-515BEDA42DE2}
DEFINE_GUID(IID_HAV_IEncoder ,
	0xf2a728bd, 0xe4d3, 0x4fdd, 0xa6, 0xde, 0x51, 0x5b, 0xed, 0xa4, 0x2d, 0xe2);

struct ENCODER_DESC
{
	HVCodec codec;
	HVEncoderPreset preset;
	unsigned int encoded_width;
	unsigned int encoded_height;
	unsigned int max_encoded_width;
	unsigned int max_encoded_height;
	unsigned int framerate_num;
	unsigned int framerate_den;
	unsigned int num_in_back_buffers;
	unsigned int num_out_back_buffers;
};

class __declspec(uuid("E4EA258B-86C8-4528-9AFE-C1DCE01ACB1A")) IEncoder : public IHAVComponent
{
public:
	virtual winrt::hresult IsSupported(VIDEO_SOURCE_DESC desc) = 0;
	virtual winrt::hresult Encode(IFrame* inFrame) = 0;
	virtual winrt::hresult GetEncodedPacket(IPacket* outPacket) = 0;
	virtual winrt::hresult GetSequenceParams(unsigned int* size, void** spps) = 0;
};