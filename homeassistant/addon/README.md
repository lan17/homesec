# HomeSec Home Assistant Add-on

This repository provides the HomeSec add-on for Home Assistant OS/Supervised.

## Install

1. In Home Assistant, open **Settings → Add-ons → Add-on Store**.
2. Add this repository: `https://github.com/lan17/homesec`.
3. Install the **HomeSec** add-on and start it.

## Notes

- The add-on generates `/config/homesec/config.yaml` on first start.
- After initial bootstrap, configuration should be managed via the REST API or
  the Home Assistant integration.
