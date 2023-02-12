#include "HAVUtilsPrivate.hpp"

HVFormat DXGIFmtHV(DXGI_FORMAT dxgi_format)
{

	switch (dxgi_format)
	{
	case DXGI_FORMAT_UNKNOWN:
		break;
	case DXGI_FORMAT_R16G16B16A16_FLOAT:
		return HV_FORMAT_BGRA64_HDR10;
	case DXGI_FORMAT_R8G8B8A8_UINT:
	case DXGI_FORMAT_B8G8R8A8_UNORM:
		return HV_FORMAT_BGRA32;
		break;
	default:
		return HV_FORMAT_UNKNOWN;
		break;
	}
}

HVColorSpace DXGICsHV(DXGI_COLOR_SPACE_TYPE dxgi_colorspace)
{
	switch (dxgi_colorspace)
	{
	case DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709:
	case DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709:
		return HV_COLORSPACE_BT709;
		break;

	case DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020:
	case DXGI_COLOR_SPACE_RGB_STUDIO_G22_NONE_P2020:
	case DXGI_COLOR_SPACE_RGB_STUDIO_G2084_NONE_P2020:
		return HV_COLORSPACE_BT2020;
		break;

	case DXGI_COLOR_SPACE_CUSTOM:
		return HV_COLORSPACE_CUSTOM;
		break;

	default:
		return HV_COLORSPACE_CUSTOM;
		break;
	}
}
