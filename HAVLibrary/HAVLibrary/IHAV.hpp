#pragma once
#include "IDecoder.hpp"
#include "IDemuxer.hpp"
#include "IDisplay.hpp"
#include "IEncoder.hpp"
#include "IMuxer.hpp"
#include "IVideoOutput.hpp"
#include "IDev.hpp"

// {C65C1195-E8ED-4A8C-959B-12941A71FB0C}
#define CLSID_HAV_IHAV_TEXT "{c65c1195-e8eD-4a8c-959b-12941a71fb0c}"
DEFINE_GUID(CLSID_HAV_IHAV,
	0xc65c1195, 0xe8ed, 0x4a8c, 0x95, 0x9b, 0x12, 0x94, 0x1a, 0x71, 0xfb, 0xc);

// {D29BE54E-46E0-4A19-BC88-6C12673FE823}
DEFINE_GUID(IID_HAV_IHAV,
	0xd29be54e, 0x46e0, 0x4a19, 0xbc, 0x88, 0x6c, 0x12, 0x67, 0x3f, 0xe8, 0x23);

class __declspec(uuid("D29BE54E-46E0-4A19-BC88-6C12673FE823")) IHAV : public IUnknown
{
public:
	virtual winrt::hresult STDMETHODCALLTYPE Link(IHAVComponent* In, IHAVComponent* Out) = 0;
	virtual winrt::hresult STDMETHODCALLTYPE CreateDevice(REFIID iid, DEV_DESC dev_desc, IDev** Out) = 0;
	virtual winrt::hresult STDMETHODCALLTYPE CreateDemuxer(REFIID iid, IDemuxer **Out) = 0;
	virtual winrt::hresult STDMETHODCALLTYPE CreateMuxer(REFIID iid, IMuxer **Out) = 0;
	virtual winrt::hresult STDMETHODCALLTYPE CreateDisplay(REFIID iid, IDisplay** Out) = 0;
	virtual winrt::hresult STDMETHODCALLTYPE CreateDecoder(REFIID iid, IDecoder **Out) = 0;
	virtual winrt::hresult STDMETHODCALLTYPE CreateEncoder(REFIID iid, ENCODER_DESC encoder_desc, IDev* dev, IEncoder** Out) = 0;
	virtual winrt::hresult STDMETHODCALLTYPE CreateFrame(REFIID iid, FRAME_OUTPUT_DESC frame_desc, IFrame** Out) = 0;
	virtual winrt::hresult STDMETHODCALLTYPE CreatePacket(REFIID iid, IPacket** Out) = 0;
};
