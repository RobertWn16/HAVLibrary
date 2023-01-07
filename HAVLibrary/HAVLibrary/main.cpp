#include "FFMPEGDemuxer.hpp"

int main(int argc, char** argv)
{
	winrt::com_ptr<IDemuxer> nv = winrt::make_self<FFMPEGDemuxer>();
	nv.get()->VideoCapture("essa");
	nv.get()->VideoCapture(0);

	return 0;
}