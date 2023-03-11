#pragma once
#include "IVideoOutput.hpp"

// {0BD7E894-CB53-4EBD-A151-897A5761C1F9}
DEFINE_GUID(IID_HAV_IMuxer,
	0xbd7e894, 0xcb53, 0x4ebd, 0xa1, 0x51, 0x89, 0x7a, 0x57, 0x61, 0xc1, 0xf9);

class __declspec(uuid("0BD7E894-CB53-4EBD-A151-897A5761C1F9")) IMuxer : public IHAVComponent
{
public:
	virtual winrt::hresult VideoStream(std::string path, VIDEO_OUTPUT_DESC outDesc, IVideoOutput** out) = 0;
};
