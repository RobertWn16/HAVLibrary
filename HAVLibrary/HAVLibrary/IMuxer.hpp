#include "IVideoOutput.hpp"

class __declspec(uuid("D31D1F45-A653-4668-BE4B-A074668FA9CD")) IMuxer : public IHAVComponent
{
	virtual winrt::hresult VideoStream(std::string path, VIDEO_OUTPUT_DESC outDesc, IVideoOutput** out) = 0;
};
