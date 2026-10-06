-- Domain model. Names follow the blueprint (deal, company, financial_metric, document, risk, synergy, system, dependency,
-- integration_workstream, milestone, decision, evidence, model_artifact); additions are marked (+).

CREATE TABLE deal (deal_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, external_ref text, name text, metadata jsonb NOT NULL DEFAULT '{}', created_at timestamptz NOT NULL DEFAULT now(),
  target_name text NOT NULL, price numeric NOT NULL CHECK (price > 0), standalone_value numeric NOT NULL CHECK (standalone_value > 0), acquirer_revenue numeric NOT NULL CHECK (acquirer_revenue > 0),   -- (+) the price and what the target is worth alone
  management_case jsonb NOT NULL, relatedness double precision NOT NULL CHECK (relatedness BETWEEN 0 AND 1), seed int, UNIQUE (tenant_id, external_ref));                                   -- (+) the announced synergies; seed of the synthetic data room
SELECT enable_tenant_rls('deal');

-- Every organisation in the entity graph: the two parties and both sides' customers.
CREATE TABLE company (company_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, deal_id uuid REFERENCES deal, status text NOT NULL CHECK (status IN ('acquirer', 'target', 'target_customer', 'acquirer_customer')),
  attributes jsonb NOT NULL DEFAULT '{}', observed_at timestamptz, created_at timestamptz NOT NULL DEFAULT now(), name text NOT NULL, revenue numeric);   -- (+) name, revenue with the target
SELECT enable_tenant_rls('company');
CREATE INDEX company_deal ON company (deal_id, status);

CREATE TABLE entity_edge (tenant_id uuid NOT NULL REFERENCES tenants, deal_id uuid NOT NULL REFERENCES deal, src uuid NOT NULL REFERENCES company, dst uuid NOT NULL REFERENCES company,   -- (+) the graph's edges
  relation text NOT NULL CHECK (relation IN ('same_as')), confidence double precision NOT NULL, method text NOT NULL, PRIMARY KEY (src, dst, relation));
SELECT enable_tenant_rls('entity_edge');

CREATE TABLE financial_metric (tenant_id uuid NOT NULL REFERENCES tenants, deal_id uuid NOT NULL REFERENCES deal, period text NOT NULL, metric text NOT NULL, value numeric NOT NULL, PRIMARY KEY (deal_id, period, metric));
SELECT enable_tenant_rls('financial_metric');

CREATE TABLE document (document_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, deal_id uuid NOT NULL REFERENCES deal, ref text NOT NULL, kind text NOT NULL, title text NOT NULL,
  body text NOT NULL, sha256 text NOT NULL, company_id uuid REFERENCES company, created_at timestamptz NOT NULL DEFAULT now(), UNIQUE (deal_id, ref));
SELECT enable_tenant_rls('document');

CREATE TABLE risk (risk_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, deal_id uuid NOT NULL REFERENCES deal, kind text NOT NULL, severity text NOT NULL CHECK (severity IN ('high', 'medium', 'low')),
  title text NOT NULL, exposure_usd numeric, confidence double precision, detail jsonb NOT NULL DEFAULT '{}', status text NOT NULL DEFAULT 'open', created_at timestamptz NOT NULL DEFAULT now());
SELECT enable_tenant_rls('risk');

-- Where a risk, a synergy figure or a decision came from: a document and the sentence in it, or a model version.
CREATE TABLE evidence (evidence_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, deal_id uuid NOT NULL REFERENCES deal, subject_type text NOT NULL, subject_id uuid NOT NULL,
  document_id uuid REFERENCES document, locator text, snippet text, model_id uuid, created_at timestamptz NOT NULL DEFAULT now());
SELECT enable_tenant_rls('evidence');
CREATE INDEX evidence_subject ON evidence (subject_id);

-- (+) The licensed deal database the synergy and ERP models learn from.
CREATE TABLE comparable_deal (tenant_id uuid NOT NULL REFERENCES tenants, ref text NOT NULL, months_since_close int NOT NULL, still_disclosing boolean NOT NULL, features jsonb NOT NULL, reports jsonb NOT NULL, erp jsonb, PRIMARY KEY (tenant_id, ref));
SELECT enable_tenant_rls('comparable_deal');

CREATE TABLE model_artifact (model_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, model_name text NOT NULL, version text NOT NULL, artifact_uri text NOT NULL, feature_schema jsonb NOT NULL,
  metrics jsonb NOT NULL DEFAULT '{}', approved boolean NOT NULL DEFAULT false, created_at timestamptz NOT NULL DEFAULT now(), params jsonb NOT NULL, UNIQUE (tenant_id, model_name, version));   -- (+) params
SELECT enable_tenant_rls('model_artifact');

CREATE TABLE synergy (synergy_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, deal_id uuid NOT NULL REFERENCES deal, kind text NOT NULL CHECK (kind IN ('cost', 'revenue')),
  basis text NOT NULL CHECK (basis IN ('management_case', 'comparables', 'model', 'tracked')), approach text, p10 numeric, p50 numeric NOT NULL, p90 numeric, model_id uuid REFERENCES model_artifact, created_at timestamptz NOT NULL DEFAULT now());
SELECT enable_tenant_rls('synergy');

CREATE TABLE system (system_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, deal_id uuid NOT NULL REFERENCES deal, owner text NOT NULL CHECK (owner IN ('acquirer', 'target')), ref text NOT NULL,
  product text NOT NULL, tags text[] NOT NULL, annual_cost numeric NOT NULL, users int NOT NULL, cluster int, UNIQUE (deal_id, ref));
SELECT enable_tenant_rls('system');

CREATE TABLE dependency (tenant_id uuid NOT NULL REFERENCES tenants, deal_id uuid NOT NULL REFERENCES deal, src uuid NOT NULL REFERENCES system, dst uuid NOT NULL REFERENCES system, kind text NOT NULL, PRIMARY KEY (src, dst));
SELECT enable_tenant_rls('dependency');

CREATE TABLE integration_scenario (scenario_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, deal_id uuid NOT NULL REFERENCES deal, status text NOT NULL, params jsonb NOT NULL,   -- (+)
  metrics jsonb NOT NULL DEFAULT '{}', created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now());
SELECT enable_tenant_rls('integration_scenario');

CREATE TABLE integration_workstream (tenant_id uuid NOT NULL REFERENCES tenants, scenario_id uuid NOT NULL REFERENCES integration_scenario ON DELETE CASCADE, approach text NOT NULL, wave int NOT NULL, key text NOT NULL,
  name text NOT NULL, planned_start_week real NOT NULL, planned_finish_week real NOT NULL, PRIMARY KEY (scenario_id, approach, wave, key));
SELECT enable_tenant_rls('integration_workstream');

CREATE TABLE milestone (tenant_id uuid NOT NULL REFERENCES tenants, scenario_id uuid NOT NULL REFERENCES integration_scenario ON DELETE CASCADE, approach text NOT NULL, name text NOT NULL,
  planned_week real NOT NULL, p50_week real NOT NULL, p90_week real NOT NULL, PRIMARY KEY (scenario_id, approach, name));
SELECT enable_tenant_rls('milestone');

CREATE TABLE decision (decision_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, deal_id uuid NOT NULL REFERENCES deal, decision_type text NOT NULL, choice text NOT NULL,
  scenario_id uuid REFERENCES integration_scenario, status text NOT NULL CHECK (status IN ('proposed', 'approved')), rationale text NOT NULL, inputs_hash text NOT NULL, model_ids jsonb NOT NULL,
  expected jsonb NOT NULL, policy_state jsonb NOT NULL DEFAULT '{}', proposed_by uuid NOT NULL, approved_by uuid, approved_at timestamptz, created_at timestamptz NOT NULL DEFAULT now());
SELECT enable_tenant_rls('decision');

-- (+) Quarterly synergy run-rate reports after close, for tracking realisation.
CREATE TABLE realisation_report (tenant_id uuid NOT NULL REFERENCES tenants, deal_id uuid NOT NULL REFERENCES deal, quarter int NOT NULL, month int NOT NULL, cost_run_rate numeric NOT NULL, revenue_run_rate numeric NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY (deal_id, quarter));
SELECT enable_tenant_rls('realisation_report');
