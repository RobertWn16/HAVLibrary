#pragma once
#include "IPacket.hpp"

struct FFMPEGPacket : winrt::implements<FFMPEGPacket, IPacket>
{
public:
	AVPacket* pkt;
public:
	void RecievePacket(AVPacket* inPacket);
};
