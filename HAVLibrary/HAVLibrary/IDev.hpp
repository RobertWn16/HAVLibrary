#include "IHAVComponent.hpp"

// {D763242C-A1EC-4421-9B3C-06C391831AA2}
DEFINE_GUID(IID_HAV_IDev,
	0xd763242c, 0xa1ec, 0x4421, 0x9b, 0x3c, 0x6, 0xc3, 0x91, 0x83, 0x1a, 0xa2);

enum Vendor
{
	Intel = 0,
	NVIDIA = 0x10DE,
	AMD = 1
};

struct DEV_DESC
{
	unsigned int ordinal;
	Vendor vendor;
	std::string DeviceName;
};

class __declspec(uuid("D763242C-A1EC-4421-9B3C-06C391831AA2")) IDev : public IHAVComponent
{
public:
	virtual winrt::hresult GetDesc(DEV_DESC &desc) = 0;
};
