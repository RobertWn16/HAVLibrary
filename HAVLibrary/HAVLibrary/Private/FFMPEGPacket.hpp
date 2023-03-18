#pragma once
#include "IPacket.hpp"
#include "HAVUtilsPrivate.hpp"

struct FFMPEGPacket : winrt::implements<FFMPEGPacket, IPacket>
{
public:
	AVPacket* ffmpegPack;
public:
	virtual winrt::hresult GetDesc(PACKET_DESC& desc);
	void RecievePacket(AVPacket* inPacket);
	AVPacket* GetPacket();
};
