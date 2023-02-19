#pragma once
#include "IMuxer.hpp"
#include "FFMPEGPacket.hpp"
#include "HAVUtilsPrivate.hpp"

struct FFMPEGMuxer : winrt::implements<FFMPEGMuxer, IMuxer>
{
	AVIOContext* client = NULL, * server = NULL;
	AVFormatContext* oc = NULL;
	AVStream* vs = NULL;
	AVPacket* pkt;
	AVStream* st;
	FILE* out;
public:
	~FFMPEGMuxer();
	winrt::hresult VideoStream();
	winrt::hresult Stream(IPacket* inPkt);
};