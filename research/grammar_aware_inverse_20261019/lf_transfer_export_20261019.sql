-- Read-only source manifest for GAI LF-T1. Execute on research-agent Supabase.
-- Export the single 'rows' JSON array to a local file data.json (do not export this database's other contents).
-- Corpus: vms_directional_token_v01 joined to canonical vms_folios.
SELECT json_agg(
 json_build_array(d.quire,d.folio,split_part(d.paragraph_id,':p',2)::int,d.pos,d.token)
 ORDER BY d.quire,d.folio,split_part(d.paragraph_id,':p',2)::int,d.pos
) AS rows
FROM public.vms_directional_token_v01 d
JOIN vms_catalogue.vms_folios c ON c.folio_canonical=d.folio
WHERE c.section='Herbal-A'
 AND d.illustration='H'
 AND d.currier='A'
 AND d.davis_hand='1'
 AND d.token IS NOT NULL;
