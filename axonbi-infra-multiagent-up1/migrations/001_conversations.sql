-- conversation_store.py creates this itself on its first write; run it by hand
-- only if the database user may not create tables.
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id  text PRIMARY KEY,
    client_id        text NOT NULL,
    session_id       text NOT NULL,
    phone            text,
    started_at       timestamptz NOT NULL,
    last_message_at  timestamptz NOT NULL,
    turns            integer NOT NULL DEFAULT 0,
    intents          text[] NOT NULL DEFAULT '{}',
    last_agent       text,
    outcome          text,
    doctor_name      text,
    branch_name      text,
    appointment_at   text,
    booking_ref      text,
    escalated        boolean NOT NULL DEFAULT false,
    llm_calls        integer NOT NULL DEFAULT 0,
    tokens           integer NOT NULL DEFAULT 0,
    details          jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS conversations_client_last ON conversations (client_id, last_message_at DESC);
CREATE INDEX IF NOT EXISTS conversations_phone ON conversations (phone);
CREATE INDEX IF NOT EXISTS conversations_outcome ON conversations (client_id, outcome);
