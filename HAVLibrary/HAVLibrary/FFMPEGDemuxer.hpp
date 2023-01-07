#include "FFMPEGCaptureSource.hpp"
#include <vector>

struct FVContext;
struct FFMPEGDemuxer : winrt::implements<FFMPEGDemuxer, IDemuxer>
{
public:
	winrt::hresult VideoCapture(int ordinal) final;
	winrt::hresult VideoCapture(std::string path, IVCaptureSource** source) final;
	winrt::hresult VideoCapture(std::string IP, unsigned short port) final;
	winrt::hresult DesktopCapture(int monitor) final;
};
