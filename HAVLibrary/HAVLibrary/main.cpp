#include "FFMPEGDemuxer.hpp"

int main(int argc, char** argv)
{
	VIDEO_SOURCE_DESC desc;
	winrt::com_ptr<IDemuxer> nv = winrt::make_self<FFMPEGDemuxer>();
	winrt::com_ptr<IVideoSource> source;
	nv.get()->VideoCapture("sample.mp4", source.put());
	source.get()->GetDesc(desc);

	return 0;
}