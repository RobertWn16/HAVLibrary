#include "FFMPEGPacket.hpp"

void FFMPEGPacket::RecievePacket(AVPacket* inPacket)
{
	if (pkt)
	{
		if (pkt->data)
			av_packet_unref(pkt);
	}

	pkt = av_packet_clone(inPacket);
}
