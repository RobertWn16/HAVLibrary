#pragma once
#include "IPacket.hpp"
#include "HAVUtilsPrivate.hpp"

struct FFMPEGPacket : winrt::implements<FFMPEGPacket, IPacket>
{
public:
	AVPacket* ffmpegPack;
public:
	void RecievePacket(AVPacket* inPacket);
	AVPacket* GetPacket();
};
