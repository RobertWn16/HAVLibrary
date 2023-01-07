#include "FFMPEGDemuxer.hpp"
#include <iostream>
winrt::hresult FFMPEGDemuxer::VideoCapture(int ordinal)
{
	std::cout << "Call from video capture " << ordinal;
	return S_OK;
}

winrt::hresult FFMPEGDemuxer::VideoCapture(std::string path)
{
	std::cout << "Call from video capture " << path;
	return S_OK;
}

winrt::hresult FFMPEGDemuxer::VideoCapture(std::string IP, unsigned short port)
{
	return winrt::hresult();
}

winrt::hresult FFMPEGDemuxer::DesktopCapture(int monitor)
{
	return E_NOTIMPL;
}
