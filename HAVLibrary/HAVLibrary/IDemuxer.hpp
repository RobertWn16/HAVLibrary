#pragma once
#include "IVideoSource.hpp"
// {A5FA5612-DAB8-49E1-A067-13A12CD9D10C}
DEFINE_GUID(IID_HAV_IDemuxer,
	0xa5fa5612, 0xdab8, 0x49e1, 0xa0, 0x67, 0x13, 0xa1, 0x2c, 0xd9, 0xd1, 0xc);

class __declspec(uuid("E4EA258B-86C8-4528-9AFE-C1DCE01ACB1A")) IDemuxer : public IUnknown
{
public:
	virtual winrt::hresult VideoCapture(int ordinal) = 0;
	virtual winrt::hresult VideoCapture(std::string path, IVideoSource** source) = 0;
	virtual winrt::hresult VideoCapture(std::string IP, unsigned short port) = 0;
	virtual winrt::hresult DesktopCapture(int monitor) = 0;
};