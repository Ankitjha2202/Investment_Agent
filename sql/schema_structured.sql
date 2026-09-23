-- Step 1: structured schema only (investments + performance)
-- Meeting notes + pgvector come in a later step.

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS investments CASCADE;
CREATE TABLE investments (
    id                          BIGSERIAL PRIMARY KEY,
    client_id                   TEXT NOT NULL,
    client_name                 TEXT,
    client_group_id             INTEGER,
    deal_id                     TEXT,
    capital_call_id             INTEGER,
    investment_amount_usd       DOUBLE PRECISION,
    investment_amount_natural   DOUBLE PRECISION,
    natural_currency_code       TEXT,
    investment_exchange_rate    DOUBLE PRECISION,
    invested_date               DATE,
    deal_name                   TEXT,
    lob_code                    TEXT,
    lob_name                    TEXT,
    realised                    BOOLEAN,
    client_status               TEXT,
    account_name                TEXT,
    account_name_org            TEXT,
    account_rm                  TEXT,
    account_rm_org              TEXT,
    account_owner_id            TEXT,
    account_rm_email            TEXT,
    account_rm_email_org        TEXT,
    rm_alias                    TEXT
);

CREATE INDEX idx_investments_client_id ON investments (client_id);
CREATE INDEX idx_investments_group_id ON investments (client_group_id);
CREATE INDEX idx_investments_deal_id ON investments (deal_id);
CREATE INDEX idx_investments_rm ON investments (account_rm);
CREATE INDEX idx_investments_rm_alias ON investments (rm_alias);
CREATE INDEX idx_investments_date ON investments (invested_date);
CREATE INDEX idx_investments_lob ON investments (lob_code);

DROP TABLE IF EXISTS performance CASCADE;
CREATE TABLE performance (
    id                          BIGSERIAL PRIMARY KEY,
    client_group_id             INTEGER,
    client_id                   TEXT,
    client_name                 TEXT,
    is_group                    BOOLEAN,
    ci_cy_fr_amount             DOUBLE PRECISION,
    ci_current_irr              DOUBLE PRECISION,
    ci_fr_si_amount             DOUBLE PRECISION,
    ci_current_moic             TEXT,
    ci_l3y_dis_amount           DOUBLE PRECISION,
    ci_total_irr                DOUBLE PRECISION,
    ci_l3y_fr_amount            DOUBLE PRECISION,
    ci_total_moic               TEXT,
    hf_aum_amount               DOUBLE PRECISION,
    ci_realised_irr             DOUBLE PRECISION,
    mena_aum_amount             DOUBLE PRECISION,
    ci_realised_moic            TEXT,
    pref_shares_aum_amount      DOUBLE PRECISION,
    ci_since_2001_irr           DOUBLE PRECISION,
    re_cy_fr_amount             DOUBLE PRECISION,
    hf_total_moic               TEXT,
    re_fr_si_amount             DOUBLE PRECISION,
    hf_total_irr                DOUBLE PRECISION,
    re_l3y_dis_amount           DOUBLE PRECISION,
    re_core_irr                 DOUBLE PRECISION,
    re_l3y_fr_amount            DOUBLE PRECISION,
    re_core_moic                TEXT,
    re_aum_amount               DOUBLE PRECISION,
    re_current_moic             TEXT,
    receivables_amount          DOUBLE PRECISION,
    re_current_irr              DOUBLE PRECISION,
    tech_aum_amount             DOUBLE PRECISION,
    re_total_irr                DOUBLE PRECISION,
    as_of_date                  DATE,
    re_total_moic               TEXT,
    client_last_met_date        DATE,
    re_realised_irr             DOUBLE PRECISION,
    last_hf_investment_date     DATE,
    re_realised_moic            TEXT,
    last_ci_investment_date     DATE,
    last_ci_investment_name     TEXT,
    last_ci_investment_amount   DOUBLE PRECISION,
    last_re_investment_name     TEXT,
    last_re_investment_date     DATE,
    first_ci_investment_name    TEXT,
    last_re_investment_amount   DOUBLE PRECISION,
    first_re_investment_name    TEXT,
    first_hf_investment_date    DATE,
    investment_status_name      TEXT,
    first_ci_investment_date    DATE,
    ci_status_name              TEXT,
    first_re_investment_date    DATE,
    re_status_name              TEXT,
    call_account_balance_amount DOUBLE PRECISION,
    inf_status_name             TEXT,
    product_count_number        INTEGER,
    icm_status_name             TEXT,
    total_aum_amount            DOUBLE PRECISION,
    future_distribution_amount  DOUBLE PRECISION,
    ci_aum_amount               DOUBLE PRECISION,
    cop_total_irr               DOUBLE PRECISION,
    last_cop_investment_name    TEXT,
    cop_realised_irr            DOUBLE PRECISION,
    cop_current_irr             DOUBLE PRECISION,
    cop_fr_si_amount            DOUBLE PRECISION,
    last_cop_investment_amount  DOUBLE PRECISION
);

CREATE INDEX idx_performance_client_id ON performance (client_id);
CREATE INDEX idx_performance_group_id ON performance (client_group_id);
CREATE INDEX idx_performance_is_group ON performance (is_group);
