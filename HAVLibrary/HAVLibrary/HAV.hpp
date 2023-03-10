#include "IHAV.hpp"
#include "NVDEC.hpp"
#include "WinDisplay.hpp"
#include "DevNVIDIA.hpp"
#include "FFMPEGDemuxer.hpp"
#include "FFMPEGMuxer.hpp"
#include "NVENC.hpp"
#include "NVJpegDecoder.hpp"

struct HAV : winrt::implements<HAV, IHAV>
{
public:   
	winrt::hresult STDMETHODCALLTYPE Link(IHAVComponent* In, IHAVComponent* Out);
	winrt::hresult STDMETHODCALLTYPE CreateDevice(REFIID iid, DEV_DESC dev_desc, IDev** Out);
	winrt::hresult STDMETHODCALLTYPE CreateDemuxer(REFIID iid, IDemuxer **Out);
	winrt::hresult STDMETHODCALLTYPE CreateMuxer(REFIID iid, IMuxer** Out);
	winrt::hresult STDMETHODCALLTYPE CreateDisplay(REFIID iid, IDisplay **Out);
	winrt::hresult STDMETHODCALLTYPE CreateDecoder(REFIID iid, IDecoder **Out);
	winrt::hresult STDMETHODCALLTYPE CreateEncoder(REFIID iid, ENCODER_DESC encoder_desc, IDev* dev, IEncoder** Out);
	winrt::hresult STDMETHODCALLTYPE CreateFrame(REFIID iid, FRAME_OUTPUT_DESC frame_desc, IFrame** Out);
	winrt::hresult STDMETHODCALLTYPE CreatePacket(REFIID iid, IPacket** Out);
};