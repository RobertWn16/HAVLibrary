#pragma once
#include "IMuxer.hpp"
#include "HAVUtilsPrivate.hpp"

struct FFMPEGMuxer : winrt::implements<FFMPEGMuxer, IMuxer>
{
	AVFormatContext* oc = NULL;
	AVStream* vs = NULL;
	AVPacket* pkt;
public:
	~FFMPEGMuxer();
	winrt::hresult VideoStream();
	winrt::hresult Stream(uint8_t* data, unsigned int size, int fps);
};