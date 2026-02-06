# PostgreSQL Quick Start Guide

Fast track to get PostgreSQL running with flux-rag in 5 minutes.

## Prerequisites

- Python 3.11+
- PostgreSQL 14+ (or Docker)

## Option 1: Docker (Fastest)

### 1. Start PostgreSQL Container

```bash
docker run -d \
  --name flux-rag-postgres \
  -e POSTGRES_USER=flux_rag \
  -e POSTGRES_PASSWORD=flux_rag \
  -e POSTGRES_DB=flux_rag \
  -p 5432:5432 \
  postgres:16-alpine
```

### 2. Install Dependencies

```bash
cd backend
poetry install
```

### 3. Configure Application

Create or edit `.env`:
```bash
DATABASE_BACKEND=postgres
POSTGRES_DSN=postgresql://flux_rag:flux_rag@localhost:5432/flux_rag
```

### 4. Test Connection

```bash
python scripts/test_postgres_connection.py --check
```

### 5. Migrate Existing Data (Optional)

```bash
python scripts/migrate_sqlite_to_postgres.py
```

### 6. Start Application

```bash
uvicorn api.main:app --reload
```

**Done!** 🎉

## Option 2: Local PostgreSQL

### macOS (Homebrew)

```bash
# Install
brew install postgresql@16
brew services start postgresql@16

# Create database
createdb flux_rag

# Create user
psql postgres -c "CREATE USER flux_rag WITH PASSWORD 'flux_rag';"
psql postgres -c "GRANT ALL ON DATABASE flux_rag TO flux_rag;"

# Configure app
echo "DATABASE_BACKEND=postgres" >> .env
echo "POSTGRES_DSN=postgresql://flux_rag:flux_rag@localhost:5432/flux_rag" >> .env

# Test
python scripts/test_postgres_connection.py --check
```

### Ubuntu/Debian

```bash
# Install
sudo apt-get update
sudo apt-get install postgresql-16

# Create database and user
sudo -u postgres psql -c "CREATE DATABASE flux_rag;"
sudo -u postgres psql -c "CREATE USER flux_rag WITH PASSWORD 'flux_rag';"
sudo -u postgres psql -c "GRANT ALL ON DATABASE flux_rag TO flux_rag;"

# Configure app
echo "DATABASE_BACKEND=postgres" >> .env
echo "POSTGRES_DSN=postgresql://flux_rag:flux_rag@localhost:5432/flux_rag" >> .env

# Test
python scripts/test_postgres_connection.py --check
```

## Common Commands

### Check PostgreSQL Status

```bash
# Docker
docker ps | grep flux-rag-postgres

# macOS
brew services list | grep postgresql

# Linux
sudo systemctl status postgresql
```

### Connect to Database

```bash
# Docker
docker exec -it flux-rag-postgres psql -U flux_rag -d flux_rag

# Local
psql -U flux_rag -d flux_rag -h localhost
```

### View Tables

```sql
-- Inside psql
\dt                    -- List tables
\d sessions            -- Describe sessions table
\d messages            -- Describe messages table

SELECT COUNT(*) FROM sessions;
SELECT COUNT(*) FROM messages;
```

### Stop/Start PostgreSQL

```bash
# Docker
docker stop flux-rag-postgres
docker start flux-rag-postgres

# macOS
brew services stop postgresql@16
brew services start postgresql@16

# Linux
sudo systemctl stop postgresql
sudo systemctl start postgresql
```

## Verify Everything Works

```bash
# 1. Test connection
python scripts/test_postgres_connection.py --check

# Output should show:
# ✓ Connection successful
#   - Sessions: 0
#   - Messages: 0

# 2. Run full test suite
python scripts/test_postgres_connection.py

# 3. Run unit tests
python -m pytest tests/test_memory_pg.py -v

# 4. Start application
uvicorn api.main:app --reload

# 5. Test API
curl http://localhost:8000/api/sessions
```

## Switch Between SQLite and PostgreSQL

### Use SQLite (Development)

```bash
# .env
DATABASE_BACKEND=sqlite
```

### Use PostgreSQL (Production)

```bash
# .env
DATABASE_BACKEND=postgres
POSTGRES_DSN=postgresql://flux_rag:flux_rag@localhost:5432/flux_rag
```

No code changes needed - just update `.env` and restart!

## Migrate Data

```bash
# Migrate from SQLite to PostgreSQL
python scripts/migrate_sqlite_to_postgres.py

# With custom paths
python scripts/migrate_sqlite_to_postgres.py \
  --sqlite-path ./data/memory.db \
  --postgres-dsn postgresql://flux_rag:flux_rag@localhost:5432/flux_rag
```

## Troubleshooting

### "Connection refused"

PostgreSQL not running:
```bash
# Docker
docker start flux-rag-postgres

# macOS
brew services start postgresql@16

# Linux
sudo systemctl start postgresql
```

### "Authentication failed"

Wrong password in DSN:
```bash
# Reset password
psql -U postgres
ALTER USER flux_rag WITH PASSWORD 'new_password';

# Update .env
POSTGRES_DSN=postgresql://flux_rag:new_password@localhost:5432/flux_rag
```

### "ModuleNotFoundError: No module named 'asyncpg'"

Install dependencies:
```bash
poetry install
```

### "Database does not exist"

Create database:
```bash
createdb -U postgres flux_rag
# or
psql -U postgres -c "CREATE DATABASE flux_rag;"
```

## Production Deployment

For production, use strong passwords and SSL:

```bash
# .env (production)
DATABASE_BACKEND=postgres
POSTGRES_DSN=postgresql://flux_rag:STRONG_PASSWORD_HERE@prod-db.example.com:5432/flux_rag?sslmode=require
```

## Cleanup (Development Only)

### Remove Docker Container

```bash
docker stop flux-rag-postgres
docker rm flux-rag-postgres
```

### Remove Local PostgreSQL

```bash
# macOS
brew services stop postgresql@16
brew uninstall postgresql@16

# Linux
sudo apt-get remove postgresql-16
```

### Remove Data

```bash
# Docker (container removal removes data)
docker rm flux-rag-postgres

# Local
dropdb -U postgres flux_rag
```

## Next Steps

- Read full documentation: [POSTGRES_MIGRATION.md](./POSTGRES_MIGRATION.md)
- Check implementation summary: [../POSTGRES_IMPLEMENTATION_SUMMARY.md](../POSTGRES_IMPLEMENTATION_SUMMARY.md)
- Configure production settings
- Setup automated backups
- Enable monitoring

## Support

For detailed information:
- Full migration guide: [POSTGRES_MIGRATION.md](./POSTGRES_MIGRATION.md)
- Implementation details: [../POSTGRES_IMPLEMENTATION_SUMMARY.md](../POSTGRES_IMPLEMENTATION_SUMMARY.md)
- Test files: [../tests/test_memory_pg.py](../tests/test_memory_pg.py)
