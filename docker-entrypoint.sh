#!/bin/bash
set -e

# Load environment from mounted .env if present
if [ -f /config/.env ]; then
    set -a
    source /config/.env
    set +a
fi

# Run database migrations
echo "Running database migrations..."
if python -m alembic -c /app/alembic.ini upgrade head; then
    echo "Migrations complete."
else
    echo "Warning: Migrations failed (database may be unavailable). Continuing anyway..."
fi

# Start HomeSec
exec python -m homesec.cli "$@"
