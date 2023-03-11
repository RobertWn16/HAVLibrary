#include "HAV.hpp"
#include "HAVFactory.hpp"
#include <tchar.h>

HAVFactory* pHAVFty = winrt::make_self<HAVFactory>().detach(); //No need to free this singleton. He dies when all dll instances are unloaded
HMODULE as_hModule;
DWORD as_OutstandingObjects = 0;
DWORD as_LockCount = 0;

HRESULT WINAPI DllRegisterServer()
{
    DWORD dwLastError = ERROR_SUCCESS;
    HKEY hKey = NULL;
    DWORD dwSize = 0;
    wchar_t wszFilename[MAX_PATH];
    wchar_t wszInstallPath[MAX_PATH];

    if (!dwLastError)
    {
        if (!GetModuleFileNameW(as_hModule, wszFilename, MAX_PATH))
        {
            dwLastError = GetLastError();
        }
    }
    if (!dwLastError)
    {
        dwLastError = RegCreateKeyExW(
            HKEY_LOCAL_MACHINE,
            _T("SOFTWARE\\Classes\\CLSID\\") _T(CLSID_HAV_IHAV_TEXT) _T("\\InProcServer32"),
            0,
            NULL,
            REG_OPTION_NON_VOLATILE,
            KEY_WRITE | KEY_WOW64_64KEY,
            NULL,
            &hKey,
            NULL
        );
        if (hKey == NULL || hKey == INVALID_HANDLE_VALUE)
        {
            hKey = NULL;
        }
        if (hKey)
        {
            dwLastError = RegSetValueExW(
                hKey,
                NULL,
                0,
                REG_SZ,
                reinterpret_cast<const BYTE*>(wszFilename),
                (wcslen(wszFilename) + 1) * sizeof(wchar_t)
            );
            dwLastError = RegSetValueExW(
                hKey,
                L"ThreadingModel",
                0,
                REG_SZ,
                reinterpret_cast<const BYTE*>(L"Apartment"),
                10 * sizeof(wchar_t)
            );
            RegCloseKey(hKey);
        }
    }

    return dwLastError == 0 ? (NOERROR) : (HRESULT_FROM_WIN32(dwLastError));
}

HRESULT WINAPI DllUnregisterServer()
{
    DWORD dwLastError = ERROR_SUCCESS;
    HKEY hKey = NULL;
    DWORD dwSize = 0;
    wchar_t wszFilename[MAX_PATH];

    if (!dwLastError)
    {
        if (!GetModuleFileNameW(as_hModule, wszFilename, MAX_PATH))
        {
            dwLastError = GetLastError();
        }
    }
    if (!dwLastError)
    {
        dwLastError = RegOpenKeyW(
            HKEY_LOCAL_MACHINE,
            _T("SOFTWARE\\Classes\\CLSID\\") _T(CLSID_HAV_IHAV_TEXT),
            &hKey
        );
        if (hKey == NULL || hKey == INVALID_HANDLE_VALUE)
        {
            hKey = NULL;
        }
        if (hKey)
        {
            dwLastError = RegDeleteTreeW(
                hKey,
                0
            );
            RegCloseKey(hKey);
            if (!dwLastError)
            {
                RegDeleteKeyW(
                    HKEY_LOCAL_MACHINE,
                    _T("SOFTWARE\\Classes\\CLSID\\") _T(CLSID_HAV_IHAV_TEXT)
                );
            }
        }
    }

    return dwLastError == 0 ? (NOERROR) : (HRESULT_FROM_WIN32(dwLastError));
}

HRESULT WINAPI DllCanUnloadNow()
{
    return((as_OutstandingObjects | as_LockCount) ? S_FALSE : S_OK);
}

HRESULT WINAPI DllGetClassObject(
    REFCLSID objGuid,
    REFIID   factoryGuid,
    LPVOID* factoryHandle
)
{
    HRESULT  hr = S_OK;
    if (IsEqualCLSID(objGuid, CLSID_HAV_IHAV))
    {
        *factoryHandle = pHAVFty;
    }
    else
    {
        *factoryHandle = 0;
        hr = CLASS_E_CLASSNOTAVAILABLE;
    }

    return(hr);
}

BOOL WINAPI DllMain(
    _In_ HINSTANCE hinstDLL,
    _In_ DWORD     fdwReason,
    _In_ LPVOID    lpvReserved
)
{
    switch (fdwReason)
    {
    case DLL_PROCESS_ATTACH:
        DisableThreadLibraryCalls(hinstDLL);
        as_hModule = hinstDLL;
        break;
    case DLL_THREAD_ATTACH:
        break;
    case DLL_THREAD_DETACH:
        break;
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}
