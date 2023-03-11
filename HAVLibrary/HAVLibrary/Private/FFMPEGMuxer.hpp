#pragma once
#include "IMuxer.hpp"
#include "FFMPEGVideoOutput.hpp"
#include "HAVUtilsPrivate.hpp"

struct FFMPEGMuxer : winrt::implements<FFMPEGMuxer, IMuxer>
{
public:
	~FFMPEGMuxer();
	winrt::hresult VideoStream(std::string path, VIDEO_OUTPUT_DESC outDesc, IVideoOutput** out) final;
};