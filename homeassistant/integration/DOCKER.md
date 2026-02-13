# Running HomeSec with Home Assistant Docker

This guide is for users running Home Assistant as a Docker container (not HA OS).

If you're using **Home Assistant OS**, use the [HomeSec Add-on](../addon/homesec/DOCS.md) instead for a zero-config experience.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Your Docker Host                              │
│                                                                         │
│  ┌─────────────────────────────┐      ┌─────────────────────────────┐  │
│  │   Home Assistant Container  │      │   HomeSec All-in-One        │  │
│  │                             │      │                             │  │
│  │  ┌───────────────────────┐  │      │  ┌───────────────────────┐  │  │
│  │  │ Your existing config  │  │      │  │    HomeSec App        │  │  │
│  │  │ Your automations      │  │      │  │    (FastAPI)          │  │  │
│  │  │ Your integrations     │  │      │  └──────────┬────────────┘  │  │
│  │  └───────────────────────┘  │      │             │               │  │
│  │                             │      │  ┌──────────▼────────────┐  │  │
│  │  ┌───────────────────────┐  │ HTTP │  │   Bundled Postgres    │  │  │
│  │  │ HomeSec Integration   │◄─┼──────┼──│   (managed by s6)     │  │  │
│  │  │ (installed via HACS)  │  │      │  └───────────────────────┘  │  │
│  │  └───────────────────────┘  │      │                             │  │
│  │             │               │      │  ┌───────────────────────┐  │  │
│  │             │  Events API   │      │  │   /config volume      │  │  │
│  │             └───────────────┼──────┼─►│   /data volume        │  │  │
│  │                             │      │  └───────────────────────┘  │  │
│  └─────────────────────────────┘      └─────────────────────────────┘  │
│          :8123                                  :8080                   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Data Flow:**

```
┌──────────┐    RTSP/FTP     ┌──────────┐   homesec_alert   ┌──────────┐
│  Camera  │ ───────────────►│ HomeSec  │ ─────────────────►│   Home   │
│          │                 │          │   (HA Events API) │Assistant │
└──────────┘                 └──────────┘                   └──────────┘
                                  │                              │
                                  │ REST API                     │
                                  │ (cameras, clips, stats)      │
                                  ◄──────────────────────────────┘
                                    HomeSec Integration polls
```

---

## Quick Start

### Step 1: Create a Home Assistant Long-Lived Access Token

1. In Home Assistant, go to your profile (click your name in the sidebar)
2. Scroll to "Long-Lived Access Tokens"
3. Click "Create Token"
4. Name it `homesec` and copy the token

### Step 2: Run HomeSec Container

```bash
docker run -d \
  --name homesec \
  --restart unless-stopped \
  -e HA_URL=http://YOUR_HA_IP:8123 \
  -e HA_TOKEN=YOUR_LONG_LIVED_TOKEN \
  -v homesec-data:/data \
  -v homesec-config:/config \
  -v /path/to/clips:/media/clips \
  -p 8080:8080 \
  ghcr.io/lan17/homesec:latest
```

Replace:
- `YOUR_HA_IP` - Your Home Assistant IP address (e.g., `192.168.1.100`)
- `YOUR_LONG_LIVED_TOKEN` - The token from Step 1
- `/path/to/clips` - Where you want video clips stored

### Bootstrap Mode (Empty Config)

HomeSec can start with an empty or missing `/config/config.yaml`. In this mode:
- Only the API server runs (no pipeline, no cameras).
- The HA integration can still connect.
- The integration will enable the Home Assistant notifier automatically.

You still need to provide a full config and restart HomeSec to run the pipeline.
When you switch to a full config, remove any `bootstrap: true` flag if present.

### Step 3: Install the Integration (Git Sparse-Checkout)

If you want to install directly from git (e.g., a dev branch), use sparse checkout
so only the integration files are pulled into Home Assistant.

Run this inside the Home Assistant container (or on the host if `/config` is local):

```bash
cd /config/custom_components

git clone --no-checkout https://github.com/lan17/homesec.git homesec_git
cd homesec_git

git sparse-checkout init --cone
git sparse-checkout set homeassistant/integration/custom_components/homesec
git checkout <your-branch>

rm -rf /config/custom_components/homesec
cp -a homeassistant/integration/custom_components/homesec /config/custom_components/homesec
```

After updating the repo later:

```bash
cd /config/custom_components/homesec_git
git pull
cp -a homeassistant/integration/custom_components/homesec /config/custom_components/homesec
```

Restart Home Assistant after any update.

### Step 4: Configure the Integration

1. Go to Settings → Devices & Services → Add Integration
2. Search for "HomeSec"
3. Enter your HomeSec URL: `http://YOUR_DOCKER_HOST_IP:8080`
4. Complete the setup

---

## Docker Compose Setup

For users who prefer docker-compose, here's a complete setup:

```yaml
# docker-compose.yml
version: "3.8"

services:
  homesec:
    image: ghcr.io/lan17/homesec:latest
    container_name: homesec
    restart: unless-stopped
    environment:
      # Home Assistant connection (for sending alerts)
      HA_URL: http://homeassistant:8123
      HA_TOKEN: ${HA_TOKEN}  # Set in .env file

      # Optional: Use external Postgres instead of bundled
      # DATABASE_URL: postgresql+asyncpg://user:pass@postgres:5432/homesec
    volumes:
      - homesec-data:/data        # Postgres data + internal state
      - homesec-config:/config    # HomeSec configuration
      - ./clips:/media/clips      # Video clip storage
    ports:
      - "8080:8080"
    networks:
      - ha-network

  # Your existing Home Assistant container
  homeassistant:
    image: ghcr.io/home-assistant/home-assistant:stable
    container_name: homeassistant
    restart: unless-stopped
    volumes:
      - ./ha-config:/config
      - /etc/localtime:/etc/localtime:ro
    ports:
      - "8123:8123"
    networks:
      - ha-network

networks:
  ha-network:
    driver: bridge

volumes:
  homesec-data:
  homesec-config:
```

Create a `.env` file:

```bash
# .env
HA_TOKEN=your_long_lived_access_token_here
```

Then run:

```bash
docker-compose up -d
```

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HA_URL` | Yes* | Home Assistant URL for sending alerts |
| `HA_TOKEN` | Yes* | Long-lived access token for HA Events API |
| `DATABASE_URL` | No | External Postgres URL (uses bundled Postgres if not set) |
| `CONFIG_PATH` | No | Path to config.yaml (default: `/config/config.yaml`) |

*Required for alerts to appear in Home Assistant

### Volumes

| Path | Purpose |
|------|---------|
| `/data` | Postgres data, internal state (persist this!) |
| `/config` | HomeSec configuration files |
| `/media/clips` | Video clip storage |

### Ports

| Port | Purpose |
|------|---------|
| `8080` | HomeSec REST API and Web UI |

---

## Network Configurations

### Option A: Shared Docker Network (Recommended)

Both containers on the same Docker network can communicate by container name:

```
HomeSec → http://homeassistant:8123  (HA_URL)
HA Integration → http://homesec:8080
```

### Option B: Host Network

If using `network_mode: host`:

```
HomeSec → http://localhost:8123 or http://YOUR_IP:8123
HA Integration → http://localhost:8080 or http://YOUR_IP:8080
```

### Option C: Separate Docker Hosts

If HomeSec runs on a different machine:

```
HomeSec → http://HA_MACHINE_IP:8123
HA Integration → http://HOMESEC_MACHINE_IP:8080
```

---

## Camera Configuration

After starting HomeSec, configure your cameras by editing `/config/config.yaml` or using the REST API.

### Example config.yaml

```yaml
cameras:
  - name: front_door
    enabled: true
    source:
      backend: rtsp
      config:
        url_env: FRONT_DOOR_RTSP_URL

  - name: backyard
    enabled: true
    source:
      backend: rtsp
      config:
        url_env: BACKYARD_RTSP_URL

storage:
  backend: local
  config:
    base_path: /media/clips

notifiers:
  - backend: home_assistant
    config:
      url_env: HA_URL
      token_env: HA_TOKEN

server:
  enabled: true
  host: 0.0.0.0
  port: 8080
```

### Adding Cameras via REST API

```bash
curl -X POST http://localhost:8080/api/v1/cameras \
  -H "Content-Type: application/json" \
  -d '{
    "name": "front_door",
    "enabled": true,
    "source_backend": "rtsp",
    "source_config": {
      "url": "rtsp://user:pass@192.168.1.50:554/stream1"
    }
  }'
```

---

## Verifying the Setup

### Check HomeSec is Running

```bash
curl http://localhost:8080/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "pipeline": "running",
  "postgres": "connected",
  "cameras_online": 2
}
```

### Check Integration Connection

In Home Assistant:
1. Go to Settings → Devices & Services
2. Find HomeSec
3. Check that entities are available:
   - `sensor.homesec_cameras_online`
   - `sensor.homesec_alerts_today`
   - `binary_sensor.front_door_motion`
   - etc.

### Test Alert Flow

Trigger motion on a camera and verify:
1. HomeSec processes the clip
2. Alert appears in HA as `homesec_alert` event
3. Camera's motion binary sensor turns on
4. Motion clears after timeout (default: 30s)

---

## Troubleshooting

### HomeSec can't connect to Home Assistant

```
ERROR Home Assistant token not found in env: HA_TOKEN
```

**Fix**: Ensure `HA_TOKEN` environment variable is set correctly.

### Integration can't connect to HomeSec

```
Error communicating with HomeSec
```

**Fix**:
- Verify HomeSec is running: `docker logs homesec`
- Check network connectivity: `curl http://HOMESEC_IP:8080/api/v1/health`
- Ensure firewall allows port 8080

### No alerts appearing in Home Assistant

1. Check HomeSec logs: `docker logs homesec`
2. Verify HA URL is correct and reachable from HomeSec container
3. Test HA token:
   ```bash
   curl -H "Authorization: Bearer YOUR_TOKEN" http://HA_IP:8123/api/
   ```

### Database connection errors

If using bundled Postgres:
- Check `/data` volume is mounted and writable
- View Postgres logs: `docker exec homesec cat /data/postgres/log`

If using external Postgres:
- Verify `DATABASE_URL` format: `postgresql+asyncpg://user:pass@host:5432/dbname`
- Ensure database exists and user has permissions

---

## Comparison: Add-on vs Docker

| Feature | HA OS Add-on | Docker |
|---------|--------------|--------|
| Setup complexity | One-click | Manual |
| Supervisor integration | Yes | No |
| Zero-config auth | Yes (SUPERVISOR_TOKEN) | No (need HA token) |
| Automatic discovery | Yes | Manual URL entry |
| Updates | Via HA UI | `docker pull` |
| Best for | HA OS users | HA Container users |

---

## Next Steps

- [Configure automations](https://www.home-assistant.io/docs/automation/) based on HomeSec alerts
- Set up [notification actions](https://www.home-assistant.io/integrations/notify/) for high-risk events
- Create a [Lovelace dashboard](https://www.home-assistant.io/lovelace/) for camera monitoring
