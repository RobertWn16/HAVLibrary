#include "FFMPEGPacket.hpp"

void FFMPEGPacket::RecievePacket(AVPacket* inPacket)
{
	if (ffmpegPack)
		if (ffmpegPack->data)
			av_packet_unref(ffmpegPack);
	ffmpegPack = av_packet_clone(inPacket);
}

AVPacket* FFMPEGPacket::GetPacket()
{
	return ffmpegPack;
	return nullptr;
}
