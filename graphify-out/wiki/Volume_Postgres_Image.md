# Volume Postgres Image

> 15 nodes · cohesion 0.15

## Key Concepts

- **homesec service** (10 connections) — `docker-compose.yml`
- **postgres service** (6 connections) — `docker-compose.yml`
- **DB_DSN environment variable** (3 connections) — `docker-compose.yml`
- **docker-compose.yml** (2 connections) — `docker-compose.yml`
- **asyncpg PostgreSQL driver** (1 connections) — `docker-compose.yml`
- **./config/config.yaml volume** (1 connections) — `docker-compose.yml`
- **.env file** (1 connections) — `docker-compose.yml`
- **homesec_pgdata volume** (1 connections) — `docker-compose.yml`
- **leva/homesec:latest image** (1 connections) — `docker-compose.yml`
- **pg_isready healthcheck** (1 connections) — `docker-compose.yml`
- **postgres:16 image** (1 connections) — `docker-compose.yml`
- **/tmp/homesec-preview tmpfs** (1 connections) — `docker-compose.yml`
- **recordings volume** (1 connections) — `docker-compose.yml`
- **storage volume** (1 connections) — `docker-compose.yml`
- **yolo_cache volume** (1 connections) — `docker-compose.yml`

## Relationships

- No strong cross-community connections detected

## Source Files

- `docker-compose.yml`

## Audit Trail

- EXTRACTED: 32 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*