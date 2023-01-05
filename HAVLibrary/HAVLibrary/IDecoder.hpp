#include "pch.hpp"
#include "IMuxer.hpp"
// {E4EA258B-86C8-4528-9AFE-C1DCE01ACB1A}
DEFINE_GUID(IID_HAV_IDecoder,
	0xe4ea258b, 0x86c8, 0x4528, 0x9a, 0xfe, 0xc1, 0xdc, 0xe0, 0x1a, 0xcb, 0x1a);

class __declspec(uuid("E4EA258B-86C8-4528-9AFE-C1DCE01ACB1A")) IDecoder : public IUnknown
{
public:
	virtual winrt::hresult TestFunction() = 0;
};