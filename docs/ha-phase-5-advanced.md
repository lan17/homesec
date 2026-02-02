# Phase 5: Advanced Features

**Goal**: Premium features for power users.

**Estimated Effort**: 5-7 days

**Dependencies**: Phase 4 (HA Integration)

---

## Overview

This phase adds optional advanced features:
- Custom Lovelace card for camera view with detection overlays
- Event timeline panel
- Snapshot/thumbnail support

**Note**: This phase is optional and can be deferred or skipped.

---

## 5.1 Custom Lovelace Card

**File**: `homeassistant/integration/custom_components/homesec/www/homesec-camera-card.js`

### Interface

A custom card that displays:
- Camera snapshot/stream
- Motion and person detection badges
- Latest activity info and risk level
- Link to clip viewer

### Usage

```yaml
type: custom:homesec-camera-card
entity: camera.front_door  # or sensor.front_door_last_activity
```

### Constraints

- Must register with `customElements.define`
- Must implement `setConfig()` and `set hass()`
- Should update when related entities change
- Card size: 4 (standard camera card size)

---

## 5.2 Event Timeline Panel (Optional)

Custom panel for viewing event history with timeline visualization.

### Features

- Timeline view of alerts
- Filter by camera, risk level, date range
- Click to view clip
- Integration with HA's history

### Constraints

- Can use `/api/v1/clips?alerted=true` for alert history
- May require frontend build tooling (Lit, etc.)

---

## 5.3 Snapshot Support (Optional)

Add snapshot/thumbnail entities for each camera.

### Features

- `image.{camera}_snapshot` - Latest snapshot
- `image.{camera}_last_alert` - Thumbnail from last alert
- Periodic snapshot refresh

### Constraints

- Requires API endpoint to serve snapshots
- May need to store thumbnails during clip processing
- Consider storage implications

---

## File Changes Summary

| File | Change |
|------|--------|
| `custom_components/homesec/www/homesec-camera-card.js` | Lovelace card |
| `custom_components/homesec/www/homesec-timeline.js` | Timeline panel (optional) |
| `custom_components/homesec/image.py` | Image platform (optional) |

---

## Test Expectations

### Manual Test Cases

**Lovelace Card**
- Given card configured with camera entity, when view dashboard, then camera image displayed
- Given motion active, when view card, then motion badge highlighted
- Given new alert, when view card, then activity and risk info updated

**Timeline (if implemented)**
- Given events exist, when open timeline, then events displayed chronologically
- Given filter by camera, when applied, then only that camera's events shown

---

## Definition of Done

- [ ] Custom Lovelace card renders camera with overlays
- [ ] Card updates when entities change
- [ ] Card shows detection badges (motion, person)
- [ ] Card displays latest activity info
- [ ] (Optional) Timeline panel shows event history
- [ ] (Optional) Snapshot images work
