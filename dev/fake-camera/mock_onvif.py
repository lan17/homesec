import asyncio
import socket
import struct
import uuid
import re
from aiohttp import web
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ONVIF_HOST = "localhost"
ONVIF_PORT = 8000
RTSP_URL = "rtsp://localhost:8099/live"

WS_DISCOVERY_MULTICAST = "239.255.255.250"
WS_DISCOVERY_PORT = 3702

DEVICE_XADDR = f"http://{ONVIF_HOST}:{ONVIF_PORT}/onvif/device_service"
DEVICE_UUID = f"uuid:{uuid.uuid4()}"

DEVICE_SERVICE_XML = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
  <s:Body>
    <tds:GetServicesResponse>
      <tds:Service>
        <tds:Namespace>http://www.onvif.org/ver10/media/wsdl</tds:Namespace>
        <tds:XAddr>http://{ONVIF_HOST}:{ONVIF_PORT}/onvif/media_service</tds:XAddr>
      </tds:Service>
    </tds:GetServicesResponse>
  </s:Body>
</s:Envelope>"""

DEVICE_INFO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
  <s:Body>
    <tds:GetDeviceInformationResponse>
      <tds:Manufacturer>FakeCam</tds:Manufacturer>
      <tds:Model>MockCam-1000</tds:Model>
      <tds:FirmwareVersion>1.0.0</tds:FirmwareVersion>
      <tds:SerialNumber>FAKE-001</tds:SerialNumber>
      <tds:HardwareId>mock-hw-001</tds:HardwareId>
    </tds:GetDeviceInformationResponse>
  </s:Body>
</s:Envelope>"""

CAPABILITIES_XML = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" xmlns:tds="http://www.onvif.org/ver10/device/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema">
  <s:Body>
    <tds:GetCapabilitiesResponse>
      <tds:Capabilities>
        <tt:Media>
          <tt:XAddr>http://{ONVIF_HOST}:{ONVIF_PORT}/onvif/media_service</tt:XAddr>
        </tt:Media>
      </tds:Capabilities>
    </tds:GetCapabilitiesResponse>
  </s:Body>
</s:Envelope>"""

PROFILES_XML = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" xmlns:trt="http://www.onvif.org/ver10/media/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema">
  <s:Body>
    <trt:GetProfilesResponse>
      <trt:Profiles token="MainStream" fixed="true">
        <tt:Name>MainStream</tt:Name>
        <tt:VideoSourceConfiguration token="VSC_1">
          <tt:Name>VideoSource</tt:Name>
          <tt:UseCount>1</tt:UseCount>
          <tt:SourceToken>VS_1</tt:SourceToken>
          <tt:Bounds x="0" y="0" width="768" height="432"/>
        </tt:VideoSourceConfiguration>
        <tt:VideoEncoderConfiguration token="VEC_1">
          <tt:Name>H264</tt:Name>
          <tt:UseCount>1</tt:UseCount>
          <tt:Encoding>H264</tt:Encoding>
          <tt:Resolution>
            <tt:Width>768</tt:Width>
            <tt:Height>432</tt:Height>
          </tt:Resolution>
          <tt:RateControl>
            <tt:FrameRateLimit>12</tt:FrameRateLimit>
            <tt:BitrateLimit>765</tt:BitrateLimit>
          </tt:RateControl>
        </tt:VideoEncoderConfiguration>
      </trt:Profiles>
    </trt:GetProfilesResponse>
  </s:Body>
</s:Envelope>"""

STREAM_URI_XML = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" xmlns:trt="http://www.onvif.org/ver10/media/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema">
  <s:Body>
    <trt:GetStreamUriResponse>
      <trt:MediaUri>
        <tt:Uri>{RTSP_URL}</tt:Uri>
        <tt:InvalidAfterConnect>false</tt:InvalidAfterConnect>
        <tt:InvalidAfterReboot>false</tt:InvalidAfterReboot>
        <tt:Timeout>PT60S</tt:Timeout>
      </trt:MediaUri>
    </trt:GetStreamUriResponse>
  </s:Body>
</s:Envelope>"""


async def handle_device_service(request):
    body = await request.text()
    logger.info("Device service request: %s", body[:200])

    if "GetDeviceInformation" in body:
        return web.Response(text=DEVICE_INFO_XML, content_type="application/soap+xml")
    if "GetCapabilities" in body:
        return web.Response(text=CAPABILITIES_XML, content_type="application/soap+xml")
    if "GetServices" in body:
        return web.Response(text=DEVICE_SERVICE_XML, content_type="application/soap+xml")

    return web.Response(
        text='<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"><s:Body/></s:Envelope>',
        content_type="application/soap+xml",
    )


async def handle_media_service(request):
    body = await request.text()
    logger.info("Media service request: %s", body[:200])

    if "GetProfiles" in body:
        return web.Response(text=PROFILES_XML, content_type="application/soap+xml")
    if "GetStreamUri" in body:
        return web.Response(text=STREAM_URI_XML, content_type="application/soap+xml")

    return web.Response(
        text='<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"><s:Body/></s:Envelope>',
        content_type="application/soap+xml",
    )


def build_probe_match(message_id: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:a="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
            xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <s:Header>
    <a:MessageID>urn:uuid:{uuid.uuid4()}</a:MessageID>
    <a:RelatesTo>{message_id}</a:RelatesTo>
    <a:To>http://schemas.xmlsoap.org/ws/2004/08/addressing/role/anonymous</a:To>
    <a:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/ProbeMatches</a:Action>
  </s:Header>
  <s:Body>
    <d:ProbeMatches>
      <d:ProbeMatch>
        <a:EndpointReference>
          <a:Address>{DEVICE_UUID}</a:Address>
        </a:EndpointReference>
        <d:Types>dn:NetworkVideoTransmitter tds:Device</d:Types>
        <d:Scopes>onvif://www.onvif.org/type/video_encoder onvif://www.onvif.org/name/FakeCam onvif://www.onvif.org/location/Replit</d:Scopes>
        <d:XAddrs>{DEVICE_XADDR}</d:XAddrs>
        <d:MetadataVersion>1</d:MetadataVersion>
      </d:ProbeMatch>
    </d:ProbeMatches>
  </s:Body>
</s:Envelope>"""


class WSDiscoveryProtocol(asyncio.DatagramProtocol):
    def __init__(self):
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        try:
            message = data.decode("utf-8", errors="replace")
            if "Probe" not in message:
                return

            msg_id_match = re.search(r"<\w*:?MessageID[^>]*>(.*?)</\w*:?MessageID>", message)
            message_id = msg_id_match.group(1) if msg_id_match else f"urn:uuid:{uuid.uuid4()}"

            logger.info("WS-Discovery Probe from %s (MessageID: %s)", addr, message_id)

            response = build_probe_match(message_id)
            self.transport.sendto(response.encode("utf-8"), addr)
            logger.info("Sent ProbeMatch to %s", addr)
        except Exception:
            logger.exception("Error handling WS-Discovery probe")


async def start_ws_discovery():
    loop = asyncio.get_event_loop()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except AttributeError:
        pass
    sock.bind(("", WS_DISCOVERY_PORT))

    group = socket.inet_aton(WS_DISCOVERY_MULTICAST)
    mreq = struct.pack("4sL", group, socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    sock.setblocking(False)

    transport, protocol = await loop.create_datagram_endpoint(
        WSDiscoveryProtocol, sock=sock
    )
    logger.info("WS-Discovery listener started on %s:%d", WS_DISCOVERY_MULTICAST, WS_DISCOVERY_PORT)
    return transport


async def main():
    await start_ws_discovery()

    app = web.Application()
    app.router.add_post("/onvif/device_service", handle_device_service)
    app.router.add_post("/onvif/media_service", handle_media_service)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", ONVIF_PORT)
    await site.start()
    logger.info("ONVIF HTTP server started on http://0.0.0.0:%d", ONVIF_PORT)

    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
