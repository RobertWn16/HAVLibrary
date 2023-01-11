#pragma once
#include "pch.hpp"
#include "IHAVComponent.hpp"

// {78FB1499-1525-4C9F-91F1-6239AE57A897}
DEFINE_GUID(IID_HAV_IFrame,
	0x78fb1499, 0x1525, 0x4c9f, 0x91, 0xf1, 0x62, 0x39, 0xae, 0x57, 0xa8, 0x97);

class __declspec(uuid("78FB1499-1525-4C9F-91F1-6239AE57A897")) IFrame : public IUnknown
{
public:

};