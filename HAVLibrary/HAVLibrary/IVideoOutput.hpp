#pragma once
#include "IPacket.hpp"

// {C251F0E6-FECC-4D94-A9AF-6FBEB6A58048}
DEFINE_GUID(IID_HAV_IVideoOutput,
	0xc251f0e6, 0xfecc, 0x4d94, 0xa9, 0xaf, 0x6f, 0xbe, 0xb6, 0xa5, 0x80, 0x48);

struct VIDEO_OUTPUT_DESC
{
	unsigned int width;
	unsigned int height;
	HVCodec codec;
	HVContainer container;
};
class __declspec(uuid("D31D1F45-A653-4668-BE4B-A074668FA9CD")) IVideoOutput : public IHAVComponent
{
public:
	virtual winrt::hresult Write(IPacket* inPck) = 0;
};
