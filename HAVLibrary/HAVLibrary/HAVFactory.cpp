#include "HAVFactory.hpp"
#include "HAV.hpp"

HRESULT __stdcall HAVFactory::CreateInstance(IUnknown* pUnkOuter, REFIID riid, void** ppvObject) noexcept
{
    if (pUnkOuter) return CLASS_E_NOAGGREGATION;
    if (IsEqualIID(riid, IID_HAV_IHAV)) {
        try
        {
            winrt::check_pointer(ppvObject);
            winrt::com_ptr<HAV> hav_instance = winrt::make_self<HAV>();
            *ppvObject = reinterpret_cast<void**>(hav_instance.get());
            hav_instance.detach();
        } catch (winrt::hresult_error const& err){
            return err.code();
        }
    }
    return S_OK;
}

HRESULT __stdcall HAVFactory::LockServer(BOOL fLock) noexcept
{
    return S_OK;
}
