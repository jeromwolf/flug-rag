# PostgreSQL Migration Guide

This guide explains how to migrate from SQLite to PostgreSQL for the flux-rag conversation memory system.

## Overview

The flux-rag project now supports both SQLite (development) and PostgreSQL (production) for conversation memory storage. The implementation provides:

- **Same interface**: Both backends implement identical methods
- **Factory pattern**: Automatic backend selection via configuration
- **Migration script**: Easy data transfer from SQLite to PostgreSQL
- **Connection pooling**: Efficient PostgreSQL connection management (min: 2, max: 10)
- **JSONB support**: Native PostgreSQL JSON storage for metadata

## Files Added

### 1. Backend Implementation
- **`backend/agent/memory_pg.py`**: PostgreSQL memory backend
- **`backend/agent/memory_factory.py`**: Factory function for backend selection
- **`backend/scripts/migrate_sqlite_to_postgres.py`**: Migration script

### 2. Configuration
- **`backend/config/settings.py`**: Added PostgreSQL settings

### 3. Tests
- **`backend/tests/test_memory_pg.py`**: Unit tests for PostgreSQL backend

### 4. Dependencies
- **`backend/pyproject.toml`**: Added `asyncpg` dependency

## Configuration

### Settings (config/settings.py)

```python
# Database - SQLite (default)
database_url: str = "sqlite+aiosqlite:///./data/sqlite.db"

# Database - PostgreSQL (production)
postgres_dsn: str = "postgresql://flux_rag:flux_rag@localhost:5432/flux_rag"
database_backend: str = "sqlite"  # "sqlite" or "postgres"
```

### Environment Variables (.env)

```bash
# Use SQLite (development)
DATABASE_BACKEND=sqlite

# Use PostgreSQL (production)
DATABASE_BACKEND=postgres
POSTGRES_DSN=postgresql://username:password@host:port/database
```

## PostgreSQL Setup

### 1. Install PostgreSQL

**macOS (Homebrew):**
```bash
brew install postgresql@16
brew services start postgresql@16
```

**Ubuntu/Debian:**
```bash
sudo apt-get install postgresql-16
sudo systemctl start postgresql
```

**Docker:**
```bash
docker run -d \
  --name flux-rag-postgres \
  -e POSTGRES_USER=flux_rag \
  -e POSTGRES_PASSWORD=flux_rag \
  -e POSTGRES_DB=flux_rag \
  -p 5432:5432 \
  postgres:16-alpine
```

### 2. Create Database and User

```sql
-- Connect as postgres superuser
psql -U postgres

-- Create user
CREATE USER flux_rag WITH PASSWORD 'flux_rag';

-- Create database
CREATE DATABASE flux_rag OWNER flux_rag;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE flux_rag TO flux_rag;
```

### 3. Verify Connection

```bash
psql -U flux_rag -d flux_rag -h localhost

# Inside psql:
\dt  # List tables (should be empty initially)
\q   # Quit
```

## Migration Process

### Step 1: Install Dependencies

```bash
cd backend
poetry install
```

This installs `asyncpg` which is required for PostgreSQL support.

### Step 2: Run Migration Script

```bash
# Basic usage (uses defaults from settings)
python scripts/migrate_sqlite_to_postgres.py

# With custom paths
python scripts/migrate_sqlite_to_postgres.py \
  --sqlite-path ./data/memory.db \
  --postgres-dsn postgresql://flux_rag:flux_rag@localhost:5432/flux_rag \
  --batch-size 100
```

**Script Output:**
```
Starting migration from ./data/memory.db to PostgreSQL...
Reading sessions from SQLite...
Found 15 sessions
Migrating sessions...
  Migrated 10/15 sessions
Completed session migration: 15 sessions
Migrating messages...
  Migrated 100 messages...
  Migrated 200 messages...
Completed message migration: 247 messages

Verification:
  SQLite:     15 sessions, 247 messages
  PostgreSQL: 15 sessions, 247 messages

✓ Migration completed successfully!
```

### Step 3: Update Configuration

Edit `.env`:
```bash
DATABASE_BACKEND=postgres
POSTGRES_DSN=postgresql://flux_rag:flux_rag@localhost:5432/flux_rag
```

Or update `config/settings.py` directly.

### Step 4: Test Application

```bash
# Start the backend
uvicorn api.main:app --reload

# Test API endpoints
curl http://localhost:8000/api/sessions
```

## Schema Comparison

| Feature | SQLite | PostgreSQL |
|---------|--------|------------|
| ID Column | `TEXT PRIMARY KEY` | `VARCHAR(36) PRIMARY KEY` |
| Timestamp | `TEXT` (ISO format) | `TIMESTAMP` (native) |
| Metadata | `TEXT` (JSON string) | `JSONB` (native) |
| Connection | File-based | Connection pool |
| Parameters | `?` placeholders | `$1, $2, ...` placeholders |

### Tables Schema

**sessions:**
```sql
CREATE TABLE sessions (
    id VARCHAR(36) PRIMARY KEY,
    title TEXT DEFAULT '',
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    metadata JSONB DEFAULT '{}'
);
```

**messages:**
```sql
CREATE TABLE messages (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX idx_messages_session ON messages(session_id, created_at);
```

## Usage in Code

### Using Factory Pattern (Recommended)

```python
from agent.memory_factory import create_memory

# Automatically creates correct backend based on settings
memory = create_memory()

# Same interface for both backends
session_id = await memory.create_session(title="Test")
await memory.add_message(session_id, "user", "Hello")
history = await memory.get_history(session_id)
```

### Direct Backend Usage

```python
# SQLite
from agent.memory import ConversationMemory
memory = ConversationMemory(db_path="./data/memory.db")

# PostgreSQL
from agent.memory_pg import PostgresConversationMemory
memory = PostgresConversationMemory(dsn="postgresql://...")
```

## API Methods

Both backends support identical methods:

| Method | Description | Returns |
|--------|-------------|---------|
| `create_session(title, metadata)` | Create new session | `session_id: str` |
| `add_message(session_id, role, content, metadata)` | Add message | `message_id: str` |
| `get_history(session_id, limit)` | Get conversation history | `list[dict]` |
| `get_sessions(limit, offset)` | Get session list | `list[dict]` |
| `get_session(session_id)` | Get single session | `dict \| None` |
| `update_session_title(session_id, title)` | Update title | `None` |
| `delete_session(session_id)` | Delete session | `None` |
| `clear_all()` | Delete all data | `None` |

PostgreSQL-specific:
| Method | Description |
|--------|-------------|
| `close()` | Close connection pool |

## Performance Considerations

### PostgreSQL Advantages
- **Connection pooling**: Efficient connection reuse (min: 2, max: 10)
- **JSONB indexing**: Fast metadata queries
- **Concurrent access**: Multiple processes can access simultaneously
- **Production-ready**: Mature, reliable, scalable

### SQLite Advantages
- **Zero configuration**: No server setup required
- **File-based**: Easy backup and portability
- **Development**: Quick setup for local development

## Testing

### Run Unit Tests

```bash
cd backend
python -m pytest tests/test_memory_pg.py -v
```

### Test Coverage

Tests cover:
- Pool initialization and table creation
- Session CRUD operations
- Message operations
- History retrieval with ordering
- Metadata JSONB handling
- Factory function backend selection

## Troubleshooting

### Connection Errors

**Error:** `could not connect to server: Connection refused`

**Solution:**
```bash
# Check if PostgreSQL is running
pg_isready -h localhost -p 5432

# Start PostgreSQL
brew services start postgresql@16  # macOS
sudo systemctl start postgresql     # Linux
docker start flux-rag-postgres      # Docker
```

### Authentication Errors

**Error:** `FATAL: password authentication failed for user "flux_rag"`

**Solution:**
```bash
# Reset password
psql -U postgres
ALTER USER flux_rag WITH PASSWORD 'new_password';

# Update .env
POSTGRES_DSN=postgresql://flux_rag:new_password@localhost:5432/flux_rag
```

### Migration Count Mismatch

**Error:** Count mismatch after migration

**Solution:**
```bash
# Check for duplicate IDs
psql -U flux_rag -d flux_rag

SELECT id, COUNT(*)
FROM sessions
GROUP BY id
HAVING COUNT(*) > 1;

SELECT id, COUNT(*)
FROM messages
GROUP BY id
HAVING COUNT(*) > 1;

# Clear and re-run migration if needed
DELETE FROM messages;
DELETE FROM sessions;
```

## Rollback to SQLite

If you need to revert to SQLite:

1. Update `.env`:
   ```bash
   DATABASE_BACKEND=sqlite
   ```

2. Restart application

3. Original SQLite data remains untouched in `./data/memory.db`

## Production Deployment

### Recommended PostgreSQL Configuration

```ini
# postgresql.conf
max_connections = 100
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 2621kB
min_wal_size = 1GB
max_wal_size = 4GB
```

### Connection String Options

```python
# Basic
postgres_dsn = "postgresql://user:pass@host:port/database"

# With SSL
postgres_dsn = "postgresql://user:pass@host:port/database?sslmode=require"

# With connection pool tuning
postgres_dsn = "postgresql://user:pass@host:port/database?application_name=flux-rag"
```

### Backup Strategy

```bash
# Dump database
pg_dump -U flux_rag -d flux_rag -f backup_$(date +%Y%m%d).sql

# Restore database
psql -U flux_rag -d flux_rag -f backup_20240101.sql
```

## Future Enhancements

Potential improvements:
- [ ] Feedback storage migration (currently uses JSONL)
- [ ] Connection pool monitoring
- [ ] Query performance metrics
- [ ] Automated backup scheduling
- [ ] Read replica support for scaling

## References

- [asyncpg Documentation](https://magicstack.github.io/asyncpg/)
- [PostgreSQL JSON Functions](https://www.postgresql.org/docs/current/functions-json.html)
- [Connection Pooling Best Practices](https://wiki.postgresql.org/wiki/Number_Of_Database_Connections)
