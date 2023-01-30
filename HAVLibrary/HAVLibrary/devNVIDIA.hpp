#pragma once
#include "IDev.hpp"
#include <cuda.h>
#include <dxgi1_6.h>

#pragma comment(lib, "dxgi")
#pragma comment(lib, "windowscodecs")

struct DevNVIDIA : winrt::implements<DevNVIDIA, IDev>
{
public:
	winrt::hresult InitDevice(DEV_DESC dev_desc);
	winrt::hresult GetDesc(DEV_DESC& desc) final;

public:
	CUdevice cuDevice;
	CUcontext cuContext;
	DEV_DESC nv_desc;
	CUdevice GetDevice();
	CUcontext GetContext();
};