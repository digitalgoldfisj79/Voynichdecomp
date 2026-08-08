-- Alpine–Venetian Corridor Programme v0.1
-- Additive only. No existing table is altered or deleted.

create extension if not exists pgcrypto;

create table if not exists public.corridor_runs (
  run_id uuid primary key default gen_random_uuid(),
  run_label text not null unique,
  protocol_version text not null default 'v0.1',
  git_branch text,
  git_commit text,
  config_sha256 text,
  protocol_sha256 text,
  rng_seed integer not null default 20260808,
  status text not null default 'built' check (status in ('built','running','done','error','underpowered','nonresolving','cancelled')),
  stage text not null default 'stage0',
  model_manifest jsonb not null default '{}'::jsonb,
  started_at timestamptz,
  finished_at timestamptz,
  notes jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.corridor_candidates (
  candidate_key text primary key,
  registry_id text references public.manuscripts(id) on update cascade on delete set null,
  title text not null,
  shelfmark text,
  holding_institution text,
  production_place_verbatim text,
  production_place_normalized text,
  place_authority text,
  geography_class text not null default 'unresolved' check (geography_class in (
    'corridor_core','corridor_buffer','control_lombardy','control_tuscany',
    'control_bavaria_swabia','control_east_alpine','unresolved'
  )),
  geography_confidence numeric,
  dating_verbatim text,
  date_start integer,
  date_end integer,
  dating_authority text,
  time_bin text check (time_bin in ('A_primary','B_antecedent','C_reception','D_late_reception') or time_bin is null),
  antecedent_eligible boolean not null default false,
  substrate text,
  language text,
  content_tags text[] not null default '{}',
  illustration_evidence text,
  illustration_status text not null default 'unknown' check (illustration_status in ('unknown','catalogue_attested','image_verified','none_verified')),
  facsimile_url text,
  iiif_manifest_url text,
  image_coverage text not null default 'unknown' check (image_coverage in ('unknown','none','partial','complete')),
  discovery_source text,
  discovery_bias text not null default 'neutral' check (discovery_bias in ('neutral','legacy_voynich_context','unknown')),
  inclusion_status text not null default 'pending' check (inclusion_status in ('pending','included','excluded','needs_review')),
  exclusion_reason text,
  metadata jsonb not null default '{}'::jsonb,
  discovered_at timestamptz not null default now(),
  verified_at timestamptz,
  updated_at timestamptz not null default now()
);

create index if not exists corridor_candidates_geo_time_idx on public.corridor_candidates(geography_class,time_bin);
create index if not exists corridor_candidates_registry_idx on public.corridor_candidates(registry_id);

create table if not exists public.corridor_pages (
  page_id uuid primary key default gen_random_uuid(),
  candidate_key text not null references public.corridor_candidates(candidate_key) on update cascade on delete cascade,
  folio text,
  seq integer,
  canvas_id text,
  iiif_image_base text,
  image_url text,
  image_width integer,
  image_height integer,
  opaque_blind_id text not null unique,
  image_sha256 text,
  triage_status text not null default 'pending' check (triage_status in ('pending','done','error','excluded')),
  triage_classes text[] not null default '{}',
  triage_model text,
  triage_payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique(candidate_key, seq)
);

create index if not exists corridor_pages_candidate_idx on public.corridor_pages(candidate_key);

create table if not exists public.corridor_objects (
  object_id uuid primary key default gen_random_uuid(),
  page_id uuid not null references public.corridor_pages(page_id) on update cascade on delete cascade,
  object_class text not null check (object_class in (
    'plant','root','flower','zodiac','star_astronomy','bath_human',
    'architecture_cartography','diagram_geometry','other_relevant'
  )),
  bbox_norm jsonb not null,
  crop_storage_path text,
  crop_sha256 text,
  normalization_variant text,
  crop_qa text check (crop_qa in ('usable','spurious','bad_crop','merge_candidate','unreviewed') or crop_qa is null),
  description jsonb,
  description_text text,
  description_model text,
  geometry_features jsonb,
  existing_herbal_object_id uuid references public.herbal_objects(id) on update cascade on delete set null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists corridor_objects_page_class_idx on public.corridor_objects(page_id,object_class);

create table if not exists public.corridor_embeddings (
  object_id uuid not null references public.corridor_objects(object_id) on update cascade on delete cascade,
  arm text not null,
  model text not null,
  model_revision text,
  input_variant text not null,
  dim integer not null,
  embedding vector,
  created_at timestamptz not null default now(),
  primary key(object_id,arm,model,input_variant)
);

create table if not exists public.corridor_control_matches (
  run_id uuid not null references public.corridor_runs(run_id) on update cascade on delete cascade,
  corridor_candidate_key text not null references public.corridor_candidates(candidate_key) on update cascade on delete cascade,
  control_candidate_key text not null references public.corridor_candidates(candidate_key) on update cascade on delete cascade,
  match_rank integer not null,
  match_distance numeric,
  match_factors jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  primary key(run_id,corridor_candidate_key,match_rank),
  unique(run_id,corridor_candidate_key,control_candidate_key)
);

create table if not exists public.corridor_scores (
  run_id uuid not null references public.corridor_runs(run_id) on update cascade on delete cascade,
  candidate_key text not null references public.corridor_candidates(candidate_key) on update cascade on delete cascade,
  object_class text not null,
  arm text not null,
  n_objects integer not null default 0,
  raw_score double precision,
  null_center double precision,
  null_scale double precision,
  calibrated_score double precision,
  score_detail jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  primary key(run_id,candidate_key,object_class,arm)
);

create table if not exists public.corridor_results (
  run_id uuid not null references public.corridor_runs(run_id) on update cascade on delete cascade,
  result_key text not null,
  analysis_family text not null,
  estimate double precision,
  ci_low double precision,
  ci_high double precision,
  p_value double precision,
  q_value double precision,
  n_corridor integer,
  n_control integer,
  verdict text,
  detail jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  primary key(run_id,result_key)
);

alter table public.corridor_runs enable row level security;
alter table public.corridor_candidates enable row level security;
alter table public.corridor_pages enable row level security;
alter table public.corridor_objects enable row level security;
alter table public.corridor_embeddings enable row level security;
alter table public.corridor_control_matches enable row level security;
alter table public.corridor_scores enable row level security;
alter table public.corridor_results enable row level security;

comment on table public.corridor_candidates is 'Preregistered staging registry for Alpine–Venetian corridor and matched-control manuscript census. Geography must represent production/origin, never holding location.';
comment on table public.corridor_pages is 'Facsimile page inventory with opaque IDs used for blind illustration triage.';
comment on table public.corridor_scores is 'Manuscript-clustered class/arm scores. Crop rows are never treated as independent inferential n.';
