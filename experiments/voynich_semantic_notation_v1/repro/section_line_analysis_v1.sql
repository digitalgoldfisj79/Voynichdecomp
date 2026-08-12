-- VSN-B2 section/line analysis, transparency reconstruction.
--
-- Provenance:
-- * Query structures recovered from PostgreSQL pg_stat_statements on 2026-08-12.
-- * PostgreSQL normalizes literal constants to $1, $2, ... in pg_stat_statements.
-- * Literal constants below are restored from query semantics/results.
-- * The 64-permutation running-text shuffle was rerun after reconstruction and
--   reproduced the previously reported null means/SDs/p99/ge_obs values exactly.
--
-- IMPORTANT: rf_edit1_pairs contains edit-path rows. Analyses below deduplicate to
-- distinct unordered token pairs first.

-- ============================================================
-- 0. Audit: edit-path rows vs distinct unordered one-edit pairs
-- ============================================================
select count(*) rows_total,
       count(distinct (least(token_a,token_b),greatest(token_a,token_b))) distinct_unordered_pairs,
       count(*)-count(distinct (least(token_a,token_b),greatest(token_a,token_b))) duplicate_paths
from voynich_semantic_notation_v1.rf_edit1_pairs;
-- Observed: 28,435 rows; 27,307 distinct unordered pairs; 1,128 duplicate paths.

-- ============================================================
-- 1. Corrected distinct-pair section edit-location census
-- ============================================================
with pair0 as (
 select least(token_a,token_b) a,
        greatest(token_a,token_b) b,
        case when count(distinct position_class)=1
             then min(position_class)
             else 'internal' end posclass
 from voynich_semantic_notation_v1.rf_edit1_pairs
 group by 1,2
), s as (
 select distinct coalesce(section,'(missing)') section,token_rf token
 from voynich_semantic_notation_v1.rf_occurrences
 where primary_eligible
), q as (
 select x.section,p.a,p.b,p.posclass
 from pair0 p
 join s x on x.token=p.a
 join s y on y.section=x.section and y.token=p.b
)
select section,count(*) pairs,
       round(avg((posclass='prefix')::int)::numeric,6) prefix,
       round(avg((posclass='internal')::int)::numeric,6) internal,
       round(avg((posclass='suffix')::int)::numeric,6) suffix
from q
group by section
order by pairs desc;

-- ============================================================
-- 2. Corrected per-line materialisation
-- ============================================================
-- This is the table used for subsequent hierarchy analyses.
-- If rerunning from scratch, drop/rename an existing table deliberately first.

create table voynich_semantic_notation_v1.line_edit_metrics_v1 as
with pair0 as (
 select least(token_a,token_b) a,greatest(token_a,token_b) b
 from voynich_semantic_notation_v1.rf_edit1_pairs
 group by 1,2
), o as (
 select coalesce(section,'(missing)') section,
        locus_id,folio,locus,line_no,coalesce(layout_family,'?') layout_family,
        occurrence_id,token_rf,
        row_number() over(partition by locus_id order by occurrence_id) rn
 from voynich_semantic_notation_v1.rf_occurrences
 where primary_eligible
), sf as (
 select section,layout_family,token_rf,count(*)::bigint f
 from o group by 1,2,3
), sn as (
 select section,layout_family,sum(f)::bigint n
 from sf group by 1,2
), bp as (
 select a.section,a.layout_family,
        sum(2.0*a.f*b.f)/(n.n*(n.n-1)) p_random_edit
 from sf a
 join pair0 e on e.a=a.token_rf
 join sf b on b.section=a.section
          and b.layout_family=a.layout_family
          and b.token_rf=e.b
 join sn n on n.section=a.section and n.layout_family=a.layout_family
 where n.n>1
 group by 1,2,n.n
), lines as (
 select section,layout_family,locus_id,
        min(folio) folio,min(locus) locus,min(line_no) line_no,
        count(*)::int n_tokens,count(distinct token_rf)::int n_unique
 from o group by 1,2,3
), adj as (
 select a.section,a.layout_family,a.locus_id,
        count(*)::int adj_opp,
        sum((e.a is not null)::int)::int adj_hits
 from o a
 join o b on b.locus_id=a.locus_id and b.rn=a.rn+1
 left join pair0 e on e.a=least(a.token_rf,b.token_rf)
                  and e.b=greatest(a.token_rf,b.token_rf)
 group by 1,2,3
), ap as (
 select a.section,a.layout_family,a.locus_id,
        count(*)::int pair_opp,
        sum((e.a is not null)::int)::int pair_hits
 from o a
 join o b on b.locus_id=a.locus_id and b.rn>a.rn
 left join pair0 e on e.a=least(a.token_rf,b.token_rf)
                  and e.b=greatest(a.token_rf,b.token_rf)
 group by 1,2,3
)
select l.*,
       coalesce(adj.adj_opp,0) adj_opp,
       coalesce(adj.adj_hits,0) adj_hits,
       coalesce(ap.pair_opp,0) pair_opp,
       coalesce(ap.pair_hits,0) pair_hits,
       bp.p_random_edit,
       case when coalesce(adj.adj_opp,0)>0
            then adj.adj_hits::float8/adj.adj_opp end adj_rate,
       case when coalesce(ap.pair_opp,0)>0
            then ap.pair_hits::float8/ap.pair_opp end pair_rate,
       case when bp.p_random_edit>0 and coalesce(adj.adj_opp,0)>0
            then (adj.adj_hits::float8/adj.adj_opp)/bp.p_random_edit end adj_enrichment,
       case when bp.p_random_edit>0 and coalesce(ap.pair_opp,0)>0
            then (ap.pair_hits::float8/ap.pair_opp)/bp.p_random_edit end pair_enrichment
from lines l
left join adj using(section,layout_family,locus_id)
left join ap using(section,layout_family,locus_id)
left join bp using(section,layout_family);

create index on voynich_semantic_notation_v1.line_edit_metrics_v1(folio,line_no);
create index on voynich_semantic_notation_v1.line_edit_metrics_v1(section,layout_family);

-- ============================================================
-- 3. Section-conditioned line clustering, all layouts
-- ============================================================
with pair0 as (
 select least(token_a,token_b) a,greatest(token_a,token_b) b
 from voynich_semantic_notation_v1.rf_edit1_pairs group by 1,2
), o0 as (
 select coalesce(section,'(missing)') section,locus_id,folio,locus,line_no,
        occurrence_id,token_rf,
        row_number() over(partition by locus_id order by occurrence_id) rn
 from voynich_semantic_notation_v1.rf_occurrences
 where primary_eligible
), sf as (
 select section,token_rf,count(*)::bigint f from o0 group by section,token_rf
), sn as (
 select section,sum(f)::bigint n from sf group by section
), bp as (
 select a.section,sum(2.0*a.f*b.f)/(n.n*(n.n-1)) p
 from sf a
 join pair0 e on e.a=a.token_rf
 join sf b on b.section=a.section and b.token_rf=e.b
 join sn n on n.section=a.section
 group by a.section,n.n
), lines as (
 select section,locus_id,min(folio) folio,min(locus) locus,min(line_no) line_no,
        count(*) n_tokens,count(distinct token_rf) n_unique
 from o0 group by section,locus_id
), adj as (
 select a.section,a.locus_id,count(*) opp,sum((e.a is not null)::int) hits
 from o0 a
 join o0 b on b.locus_id=a.locus_id and b.rn=a.rn+1
 left join pair0 e on e.a=least(a.token_rf,b.token_rf)
                  and e.b=greatest(a.token_rf,b.token_rf)
 group by a.section,a.locus_id
), allp as (
 select a.section,a.locus_id,count(*) opp,sum((e.a is not null)::int) hits
 from o0 a
 join o0 b on b.locus_id=a.locus_id and b.rn>a.rn
 left join pair0 e on e.a=least(a.token_rf,b.token_rf)
                  and e.b=greatest(a.token_rf,b.token_rf)
 group by a.section,a.locus_id
)
select l.section,count(*) lines,
       sum(coalesce(adj.opp,0)) adj_opp,
       sum(coalesce(adj.hits,0)) adj_hits,
       round((sum(coalesce(adj.hits,0))::numeric/nullif(sum(coalesce(adj.opp,0)),0)),6) adj_rate,
       sum(coalesce(allp.opp,0)) pair_opp,
       sum(coalesce(allp.hits,0)) pair_hits,
       round((sum(coalesce(allp.hits,0))::numeric/nullif(sum(coalesce(allp.opp,0)),0)),6) pair_rate,
       round(max(bp.p)::numeric,6) section_random_rate,
       round((sum(coalesce(adj.hits,0))::numeric/nullif(sum(coalesce(adj.opp,0)),0))/max(bp.p),2) adj_enrichment,
       round((sum(coalesce(allp.hits,0))::numeric/nullif(sum(coalesce(allp.opp,0)),0))/max(bp.p),2) pair_enrichment,
       sum((coalesce(allp.hits,0)>0)::int) lines_with_hit,
       round(avg((coalesce(allp.hits,0)>0)::int)::numeric,4) frac_lines_with_hit
from lines l
left join adj using(section,locus_id)
left join allp using(section,locus_id)
join bp using(section)
group by l.section
order by lines desc;

-- ============================================================
-- 4. Layout-family decomposition
-- ============================================================
with pair0 as (
 select least(token_a,token_b) a,greatest(token_a,token_b) b
 from voynich_semantic_notation_v1.rf_edit1_pairs group by 1,2
), o as (
 select coalesce(section,'(missing)') section,locus_id,coalesce(layout_family,'?') layout_family,
        occurrence_id,token_rf,
        row_number() over(partition by locus_id order by occurrence_id) rn
 from voynich_semantic_notation_v1.rf_occurrences where primary_eligible
), sf as (
 select section,layout_family,token_rf,count(*)::bigint f from o group by 1,2,3
), sn as (
 select section,layout_family,sum(f)::bigint n from sf group by 1,2
), bp as (
 select a.section,a.layout_family,sum(2.0*a.f*b.f)/(n.n*(n.n-1)) p
 from sf a
 join pair0 e on e.a=a.token_rf
 join sf b on b.section=a.section and b.layout_family=a.layout_family and b.token_rf=e.b
 join sn n on n.section=a.section and n.layout_family=a.layout_family
 where n.n>1 group by 1,2,n.n
), adj as (
 select a.section,a.layout_family,a.locus_id,count(*) opp,sum((e.a is not null)::int) hits
 from o a join o b on b.locus_id=a.locus_id and b.rn=a.rn+1
 left join pair0 e on e.a=least(a.token_rf,b.token_rf) and e.b=greatest(a.token_rf,b.token_rf)
 group by 1,2,3
), ap as (
 select a.section,a.layout_family,a.locus_id,count(*) opp,sum((e.a is not null)::int) hits
 from o a join o b on b.locus_id=a.locus_id and b.rn>a.rn
 left join pair0 e on e.a=least(a.token_rf,b.token_rf) and e.b=greatest(a.token_rf,b.token_rf)
 group by 1,2,3
), lines as (
 select section,layout_family,locus_id,count(*) n from o group by 1,2,3
)
select l.section,l.layout_family,count(*) lines,sum(l.n) tokens,
       sum(coalesce(adj.hits,0)) adj_hits,sum(coalesce(adj.opp,0)) adj_opp,
       round(sum(coalesce(adj.hits,0))::numeric/nullif(sum(coalesce(adj.opp,0)),0),5) adj_rate,
       sum(coalesce(ap.hits,0)) pair_hits,sum(coalesce(ap.opp,0)) pair_opp,
       round(sum(coalesce(ap.hits,0))::numeric/nullif(sum(coalesce(ap.opp,0)),0),5) pair_rate,
       round(max(bp.p)::numeric,5) baseline,
       round((sum(coalesce(ap.hits,0))::numeric/nullif(sum(coalesce(ap.opp,0)),0))/nullif(max(bp.p),0),2) pair_enrichment
from lines l
left join adj using(section,layout_family,locus_id)
left join ap using(section,layout_family,locus_id)
left join bp using(section,layout_family)
group by 1,2
having sum(l.n)>=30
order by 1,4 desc;

-- ============================================================
-- 5. Running-text local edit-position distribution
-- ============================================================
with pair0 as (
 select least(token_a,token_b) a,greatest(token_a,token_b) b,
        case when count(distinct position_class)=1 then min(position_class) else 'internal' end posclass
 from voynich_semantic_notation_v1.rf_edit1_pairs group by 1,2
), o as (
 select coalesce(section,'(missing)') section,locus_id,occurrence_id,token_rf
 from voynich_semantic_notation_v1.rf_occurrences
 where primary_eligible and layout_family='P'
), hits as (
 select a.section,a.locus_id,e.a,e.b,e.posclass
 from o a
 join o b on b.locus_id=a.locus_id and b.occurrence_id>a.occurrence_id
 join pair0 e on e.a=least(a.token_rf,b.token_rf) and e.b=greatest(a.token_rf,b.token_rf)
)
select section,count(*) hit_occurrences,
       round(avg((posclass='prefix')::int)::numeric,5) prefix,
       round(avg((posclass='internal')::int)::numeric,5) internal,
       round(avg((posclass='suffix')::int)::numeric,5) suffix,
       count(distinct locus_id) lines_with_hits
from hits
group by section
order by hit_occurrences desc;

-- ============================================================
-- 6. Hostile 64-permutation within-section running-text line shuffle
-- ============================================================
-- This reconstructed query was rerun on 2026-08-12 and reproduced the earlier
-- reported results exactly, including Stars null_mean=836.16, Herbal-A=429.77,
-- Pharmaceutical=83.38, Herbal-B=17.00 and Cosmological ge_obs=27.
with pair0 as (
 select least(token_a,token_b) a,greatest(token_a,token_b) b
 from voynich_semantic_notation_v1.rf_edit1_pairs group by 1,2
), base as (
 select coalesce(section,'(missing)') section,locus_id,occurrence_id,token_rf,
        row_number() over(partition by coalesce(section,'(missing)') order by locus_id,occurrence_id) slot_rank
 from voynich_semantic_notation_v1.rf_occurrences
 where primary_eligible and layout_family='P'
), perms as (
 select generate_series(1,64) perm
), shuffled as (
 select p.perm,b.section,b.locus_id,b.occurrence_id,
        row_number() over(
          partition by p.perm,b.section
          order by md5(p.perm::text||':'||b.occurrence_id::text),b.occurrence_id
        ) token_rank,
        b.token_rf
 from perms p cross join base b
), assigned as (
 select s.perm,slot.section,slot.locus_id,slot.slot_rank,s.token_rf
 from shuffled s
 join base slot on slot.section=s.section and slot.slot_rank=s.token_rank
), phits as (
 select a.perm,a.section,count(e.a)::int hits
 from assigned a
 join assigned b on b.perm=a.perm and b.locus_id=a.locus_id and b.slot_rank>a.slot_rank
 left join pair0 e on e.a=least(a.token_rf,b.token_rf) and e.b=greatest(a.token_rf,b.token_rf)
 group by a.perm,a.section
), obs as (
 select section,sum(pair_hits)::int observed
 from voynich_semantic_notation_v1.line_edit_metrics_v1
 where layout_family='P'
 group by section
)
select o.section,o.observed,
       round(avg(h.hits)::numeric,2) null_mean,
       round(stddev_pop(h.hits)::numeric,2) null_sd,
       percentile_cont(.99) within group(order by h.hits) null_p99,
       sum((h.hits>=o.observed)::int) ge_obs,
       round(((sum((h.hits>=o.observed)::int)+1)/65.0)::numeric,5) empirical_p
from obs o
join phits h using(section)
group by o.section,o.observed
order by o.observed desc;

-- ============================================================
-- 7. Rejected pre-deduplication method (DO NOT USE)
-- ============================================================
-- The first line pass joined rf_edit1_pairs directly with an OR condition:
--   ON (p.token_a=a.token_rf AND p.token_b=b.token_rf)
--   OR (p.token_b=a.token_rf AND p.token_a=b.token_rf)
-- Because rf_edit1_pairs contains 1,128 duplicate edit paths, that could inflate
-- opportunities/hits and even produce impossible-looking line pair opportunity
-- counts. Those outputs were explicitly rejected. Sections 2-6 above use pair0.
