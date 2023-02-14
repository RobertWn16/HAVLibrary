#include "IHAVComponent.hpp"
#include "HAVUtilsPrivate.hpp"
// {D31D1F45-A653-4668-BE4B-A074668FA9CD}
DEFINE_GUID(IID_HAV_IPacket,
	0xd31d1f45, 0xa653, 0x4668, 0xbe, 0x4b, 0xa0, 0x74, 0x66, 0x8f, 0xa9, 0xcd);

class __declspec(uuid("D31D1F45-A653-4668-BE4B-A074668FA9CD")) IPacket : public IHAVComponent
{
	AVPacket av_pck;
};
