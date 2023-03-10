#include"pch.hpp"

// {26D75E38-E514-4C7C-8F25-88352BC104F3}
DEFINE_GUID(IID_HAV_HAVFactory,
	0x26d75e38, 0xe514, 0x4c7c, 0x8f, 0x25, 0x88, 0x35, 0x2b, 0xc1, 0x4, 0xf3);

class __declspec(uuid("26D75E38-E514-4C7C-8F25-88352BC104F3")) HAVFactory : public winrt::implements<HAVFactory, IClassFactory>
{
	HRESULT __stdcall CreateInstance(IUnknown* pUnkOuter, REFIID riid, void** ppvObject) noexcept override;
	HRESULT __stdcall LockServer(BOOL fLock) noexcept override;
};