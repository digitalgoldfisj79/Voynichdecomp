-- Amendment 001: internal Voynich archive prior-art / novelty gate.
-- Additive only; applied to Supabase production on 2026-08-08.

alter table public.corridor_candidates add column if not exists archive_hit_count integer not null default 0;
alter table public.corridor_candidates add column if not exists archive_prior_art_status text not null default 'unscanned'
  check (archive_prior_art_status in ('unscanned','none_found','discussed','heavily_discussed','needs_review'));

create table if not exists public.corridor_archive_prior_art (
  id bigserial primary key,
  topic_key text not null,
  candidate_key text references public.corridor_candidates(candidate_key) on update cascade on delete set null,
  query_text text not null,
  source_id text not null references public.sources(source_id) on update cascade on delete cascade,
  paragraph_index integer not null,
  passage_year integer,
  passage_author text,
  passage_url text,
  source_title text,
  source_type text,
  hit_text text not null,
  relevance text not null default 'unreviewed'
    check (relevance in ('unreviewed','direct','contextual','incidental','false_positive')),
  prior_claim text,
  novelty_implication text,
  reviewed boolean not null default false,
  created_at timestamptz not null default now(),
  unique(topic_key,source_id,paragraph_index)
);

create index if not exists corridor_archive_prior_art_topic_idx
  on public.corridor_archive_prior_art(topic_key,relevance);
create index if not exists corridor_archive_prior_art_candidate_idx
  on public.corridor_archive_prior_art(candidate_key);

create table if not exists public.corridor_novelty_register (
  evidence_key text primary key,
  candidate_key text references public.corridor_candidates(candidate_key) on update cascade on delete set null,
  evidence_type text not null
    check (evidence_type in ('new_manuscript','new_image','new_documentary_link','new_transmission_link',
      'new_feature_measurement','new_negative_evidence','new_quantitative_result','replication','prior_art')),
  description text not null,
  prior_art_status text not null
    check (prior_art_status in ('unscanned','no_prior_art_found','partially_anticipated','already_discussed','replication_or_extension')),
  archive_query text,
  archive_hit_count integer not null default 0,
  source_authority text,
  evidence_date text,
  status text not null default 'candidate' check (status in ('candidate','verified','rejected','superseded')),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.corridor_archive_prior_art enable row level security;
alter table public.corridor_novelty_register enable row level security;
