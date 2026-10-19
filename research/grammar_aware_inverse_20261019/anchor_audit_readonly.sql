-- IDENT-GATE-1 independent-anchor feasibility inventory (READ ONLY)
-- Run on research-agent; query full source provenance before treating any class as a semantic label.
-- Folio-localized segmentation is not a verified plaintext gloss.
SELECT coalesce(f.section,'UNKNOWN') AS section,
       o.object_class,
       count(*) AS n_objects,
       count(DISTINCT o.folio_seq) AS n_folios,
       count(*) FILTER (WHERE o.reviewed IS TRUE) AS n_reviewed
FROM vms_catalogue.vms_objects o
LEFT JOIN vms_catalogue.vms_folios f ON f.seq=o.folio_seq
WHERE f.section IN ('Herbal-A','Herbal-B','Balneological','Stars')
GROUP BY 1,2 ORDER BY n_reviewed DESC,n_objects DESC;

SELECT 'bio_masks' AS dataset,count(*) AS n,count(DISTINCT folio_seq) AS folios FROM public.vms_bio_masks
UNION ALL SELECT 'manual_masks',count(*),count(DISTINCT folio_seq) FROM public.vms_manual_masks
UNION ALL SELECT 'stars_masks',count(*),count(DISTINCT folio_seq) FROM public.vms_stars_masks;

SELECT 'De Balneis text/picture sections' AS dataset,count(*) AS n FROM public.cmp_balneis_sections
UNION ALL SELECT 'De Balneis witness concordance',count(*) FROM public.cmp_balneis_concordance;
