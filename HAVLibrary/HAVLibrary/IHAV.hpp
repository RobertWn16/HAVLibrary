#include "IHAVComponent.hpp"
// {D29BE54E-46E0-4A19-BC88-6C12673FE823}
DEFINE_GUID(IID_HAV_IHAV,
	0xd29be54e, 0x46e0, 0x4a19, 0xbc, 0x88, 0x6c, 0x12, 0x67, 0x3f, 0xe8, 0x23);

class __declspec(uuid("D29BE54E-46E0-4A19-BC88-6C12673FE823")) IHAV : public IUnknown
{
public:
	virtual winrt::hresult Link(IHAVComponent* In, IHAVComponent* Out) = 0;
};
