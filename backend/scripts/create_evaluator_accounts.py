"""Create 10 evaluator accounts for the KOGAS AI platform demo evaluation.

Run from the backend/ directory:
    python scripts/create_evaluator_accounts.py

Idempotent: accounts that already exist are skipped (not overwritten).
At the end a formatted credential table is printed to stdout.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure backend/ is on sys.path when run from anywhere
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import aiosqlite
from passlib.context import CryptContext

from auth.models import Role, User
from auth.user_store import UserStore

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------------------------------------------------------------------
# Evaluator account definitions
# ---------------------------------------------------------------------------

EVALUATOR_ACCOUNTS = [
    {
        "id": f"evaluator-{i:02d}",
        "username": f"evaluator{i:02d}",
        "password": f"Eval2026!{i:02d}",
        "email": f"evaluator{i:02d}@demo-platform.kr",
        "full_name": f"평가위원 {i:02d}",
        "department": "평가위원",
        "role": Role.USER,
        "is_active": True,
    }
    for i in range(1, 11)
]

# Existing demo accounts whose must_change_password should be reset to False
EXISTING_DEMO_USERNAMES = ["admin", "manager", "expert", "user", "viewer", "evaluator"]


async def main() -> None:
    store = UserStore()
    # Ensure DB and tables exist
    await store._ensure_initialized()

    created: list[dict] = []
    skipped: list[str] = []

    print("=" * 60)
    print("  KOGAS AI Platform — Evaluator Account Setup")
    print("=" * 60)

    for spec in EVALUATOR_ACCOUNTS:
        username = spec["username"]
        existing = await store.get_by_username(username)

        if existing is not None:
            print(f"  [SKIP]    {username} — already exists")
            skipped.append(username)
            continue

        user = User(
            id=spec["id"],
            username=username,
            email=spec["email"],
            full_name=spec["full_name"],
            department=spec["department"],
            role=spec["role"],
            is_active=spec["is_active"],
            created_at=datetime.now(timezone.utc),
        )

        # Hash password in a thread pool to avoid blocking the event loop
        hashed = await asyncio.to_thread(pwd_context.hash, spec["password"])

        # Insert with must_change_password=0 explicitly (create_user defaults to 0
        # via the column default, but we set it explicitly for clarity)
        async with store.get_connection() as db:
            await db.execute(
                """
                INSERT OR IGNORE INTO users
                (id, username, email, full_name, department, role,
                 is_active, password_hash, created_at, last_login,
                 must_change_password)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user.id,
                    user.username,
                    user.email,
                    user.full_name,
                    user.department,
                    user.role.value,
                    1 if user.is_active else 0,
                    hashed,
                    user.created_at.isoformat(),
                    None,
                    0,  # must_change_password = False
                ),
            )
            await db.commit()

        print(f"  [CREATED] {username}")
        created.append(spec)

    # ------------------------------------------------------------------
    # Reset must_change_password for existing demo accounts
    # ------------------------------------------------------------------
    print()
    print("Updating existing demo accounts (must_change_password → False)...")

    async with store.get_connection() as db:
        placeholders = ",".join("?" * len(EXISTING_DEMO_USERNAMES))
        cursor = await db.execute(
            f"UPDATE users SET must_change_password = 0 WHERE username IN ({placeholders})",
            EXISTING_DEMO_USERNAMES,
        )
        await db.commit()
        rows_updated = cursor.rowcount

    print(f"  {rows_updated} existing account(s) updated.")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"  Created: {len(created)}  |  Skipped (already existed): {len(skipped)}")
    print("=" * 60)
    print()
    print(f"  {'ID':<15}  {'Password':<16}  {'Role':<8}  Email")
    print(f"  {'-'*15}  {'-'*16}  {'-'*8}  {'-'*35}")

    for spec in EVALUATOR_ACCOUNTS:
        status = "(new)" if spec["username"] not in skipped else "(exists)"
        print(
            f"  {spec['username']:<15}  {spec['password']:<16}  "
            f"{spec['role'].value:<8}  {spec['email']}  {status}"
        )

    print()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
