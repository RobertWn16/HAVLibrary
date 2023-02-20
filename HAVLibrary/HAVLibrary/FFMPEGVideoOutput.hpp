#pragma once
#include "IVideoOutput.hpp"
#include "FFMPEGPacket.hpp"
struct FFMPEGVideoOutput : winrt::implements<FFMPEGVideoOutput, IVideoOutput>
{
	VIDEO_OUTPUT_DESC ffmpegOutputDesc;
	AVFormatContext* ffmpegOutputContext;
	AVStream* ffmpegOutputStream;
	int ffmpegFps;
	int64_t ffmpegPts;
public:
	~FFMPEGVideoOutput();
	winrt::hresult ConfigureVideoOutput(AVFormatContext* oc, VIDEO_OUTPUT_DESC outDesc);
	winrt::hresult Write(IPacket* inPck);
};