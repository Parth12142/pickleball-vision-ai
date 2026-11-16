-- ============================================================================
-- PICKLEBALLVISUALIZER — POSTGRESQL DATABASE SCHEMA
-- Run with: psql -U postgres -d pickleball_ai -f frontend/schema.sql
-- Or set DATABASE_URL="postgresql+psycopg://user:pass@localhost:5432/pickleball_ai"
-- ============================================================================

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    full_name VARCHAR(120) NOT NULL,
    email VARCHAR(180) UNIQUE NOT NULL,
    club_or_role VARCHAR(120) DEFAULT 'Tournament Player',
    password_hash VARCHAR(256) NOT NULL,
    subscription_tier VARCHAR(32) NOT NULL DEFAULT 'paddle',
    billing_cycle VARCHAR(16) NOT NULL DEFAULT 'monthly',
    clips_uploaded_this_month INTEGER NOT NULL DEFAULT 0,
    live_clips_analyzed_this_month INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS subscription_events (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    previous_tier VARCHAR(32),
    new_tier VARCHAR(32) NOT NULL,
    billing_cycle VARCHAR(16) NOT NULL DEFAULT 'monthly',
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
