#include "FFMPEGPacket.hpp"

winrt::hresult FFMPEGPacket::GetDesc(PACKET_DESC& desc)
{
	if (ffmpegPack) {
		desc.data = ffmpegPack->data;
		desc.size = ffmpegPack->size;
		desc.timestamp = ffmpegPack->pts;
		return S_OK;
	}

	return E_POINTER;
}

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
}
