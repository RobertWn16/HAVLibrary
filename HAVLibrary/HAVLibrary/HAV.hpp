#include "IHAV.hpp"
#include "NVDEC.hpp"
#include "DevNVIDIA.hpp"
#include "FFMPEGDemuxer.hpp"

struct HAV : winrt::implements<HAV, IHAV>
{
public:   
	winrt::hresult Link(IHAVComponent* In, IHAVComponent* Out);
};