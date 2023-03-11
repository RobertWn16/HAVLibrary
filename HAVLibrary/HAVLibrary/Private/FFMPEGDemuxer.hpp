#include "IDemuxer.hpp"
#include "FFMPEGVideoSource.hpp"
#include <vector>

struct FVContext;
struct FFMPEGDemuxer : winrt::implements<FFMPEGDemuxer, IDemuxer>
{
private:
	winrt::handle PendingDemux;
public:
	winrt::hresult VideoCapture(int ordinal) final;
	winrt::hresult VideoCapture(std::string path, IVideoSource** source) final;
	winrt::hresult VideoCapture(std::string IP, unsigned short port) final;
	winrt::hresult DesktopCapture(int monitor) final;
};
