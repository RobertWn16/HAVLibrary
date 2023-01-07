#include "IDemuxer.hpp"

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavcodec/bsf.h>

struct FFMPEGDemuxer : winrt::implements<FFMPEGDemuxer, IDemuxer>
{
private:
	AVFormatContext* dem_context;
	AVInputFormat* dem_format;
	AVBSFContext* dem_bsf_context;
	AVBitStreamFilter* dem_bit_filter;
public:
	winrt::hresult VideoCapture(int ordinal) final;
	winrt::hresult VideoCapture(std::string path) final;
	winrt::hresult VideoCapture(std::string IP, unsigned short port) final;
	winrt::hresult DesktopCapture(int monitor) final;
};