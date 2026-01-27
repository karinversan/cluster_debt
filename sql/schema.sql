CREATE TABLE IF NOT EXISTS customer_features (
  customer_id        BIGINT,
  event_timestamp    TIMESTAMP,
  balance            DOUBLE PRECISION,
  balance_frequency  DOUBLE PRECISION,
  purchases          DOUBLE PRECISION,
  oneoff_purchases   DOUBLE PRECISION,
  installment_purchases DOUBLE PRECISION,
  cash_advance       DOUBLE PRECISION,
  purchases_frequency DOUBLE PRECISION,
  oneoff_purchases_frequency DOUBLE PRECISION,
  purchases_installments_frequency DOUBLE PRECISION,
  cash_advance_frequency DOUBLE PRECISION,
  cash_advance_trx   DOUBLE PRECISION,
  purchases_trx      DOUBLE PRECISION,
  credit_limit       DOUBLE PRECISION,
  payments           DOUBLE PRECISION,
  minimum_payments   DOUBLE PRECISION,
  prc_full_payment   DOUBLE PRECISION,
  tenure             DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS customer_segments (
  customer_id      BIGINT,
  event_timestamp  TIMESTAMP,
  segment_id       INT,
  segment_name     TEXT,
  model_name       TEXT,
  model_version    TEXT,
  run_id           TEXT
);
