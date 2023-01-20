#pragma once
#include <Unknwn.h>
#include <Windows.h>
#include <Unknwn.h>
#include <winrt/base.h>
#include <winrt/windows.foundation.h>
#include <queue>
#include <initguid.h>
#include <iostream>

// {B4016E7D-1222-4B3A-B1FC-E228F7EF181D}
DEFINE_GUID(IID_HAV_NVDev,
	0xb4016e7d, 0x1222, 0x4b3a, 0xb1, 0xfc, 0xe2, 0x28, 0xf7, 0xef, 0x18, 0x1d);

DEFINE_GUID(IID_HAV_NVDEC,
	0x8e1476eb, 0xb019, 0x491e, 0x82, 0x45, 0x91, 0xcc, 0x75, 0x49, 0x78, 0x63);

// {0847C497-60DB-477F-A27C-F9CE4DFD908D}
DEFINE_GUID(IID_HAV_NVFrame,
	0x847c497, 0x60db, 0x477f, 0xa2, 0x7c, 0xf9, 0xce, 0x4d, 0xfd, 0x90, 0x8d);

// {C25DDDE7-6FFE-4CEB-8DEF-48D80C2455CD}
DEFINE_GUID(IID_HAV_FFMPEGDemuxer,
	0xc25ddde7, 0x6ffe, 0x4ceb, 0x8d, 0xef, 0x48, 0xd8, 0xc, 0x24, 0x55, 0xcd);

