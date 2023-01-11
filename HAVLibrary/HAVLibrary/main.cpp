#include "HAV.hpp"

int main(int argc, char** argv)
{
	VIDEO_SOURCE_DESC desc;
	winrt::com_ptr<IDemuxer> nv = winrt::make_self<FFMPEGDemuxer>();
	winrt::com_ptr<IVideoSource> source;
	winrt::com_ptr<IDecoder> dec = winrt::make_self<NVDEC>();
	winrt::com_ptr<DevNVIDIA> dev_nvidia = winrt::make_self<DevNVIDIA>();
	winrt::com_ptr<HAV> hav_instance = winrt::make_self<HAV>();

	DEV_DESC dev_desc;
	dev_desc.vendor = NVIDIA;
	dev_desc.ordinal = 0;

	winrt::check_hresult(dev_nvidia->InitDevice(dev_desc));

	winrt::check_hresult(hav_instance->Link(dev_nvidia.get(), dec.get()));

	winrt::check_hresult(nv->VideoCapture("sample.mp4", source.put()));
	winrt::check_hresult(source->GetDesc(desc));
	if (SUCCEEDED(dec->IsSupported(desc))) {
		winrt::check_hresult(hav_instance->Link(source.get(), dec.get()));
	}

	while (true)
	{
		dec->Decode(nullptr);
		Sleep(1000);
	}


	return 0;
}