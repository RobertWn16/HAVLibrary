#include "IDisplay.hpp"
#include "HAVUtilsPrivate.hpp"
#include "WinDisplayVideoSource.hpp"

struct WinDisplay : winrt::implements<WinDisplay, IDisplay>
{
private:
	DISPLAY_DESC display_desc;
	winrt::com_ptr<ID3D11Device> pwdDevice;
	winrt::com_ptr<ID3D11DeviceContext> pwdDeviceCtx;
	winrt::com_ptr<IDXGIOutput6> pwdOutput6;
public:
	winrt::hresult ConfigureDisplay() noexcept;
	winrt::hresult DisplayCapture(IVideoSource** out) noexcept final;
	winrt::hresult GetDesc(DISPLAY_DESC &desc) final;
};


