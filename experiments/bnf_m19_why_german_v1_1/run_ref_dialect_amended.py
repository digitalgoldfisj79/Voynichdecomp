#!/usr/bin/env python3
import urllib.request, xml.etree.ElementTree as ET
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/a98f13ce447e963c124f218a09d63352e7ac81b8/experiments/bnf_m19_why_german_v1_1/run_ref_dialect.py'
src=urllib.request.urlopen(BASE,timeout=90).read().decode('utf-8')
src=src.rsplit("if __name__=='__main__': main()",1)[0]
lib={'__name__':'ref_lib'};exec(compile(src,'run_ref_dialect.py','exec'),lib)

def parse_xml_token_trans(blob):
    try: root=ET.fromstring(blob)
    except Exception: return []
    toks=[]
    # The archived ReF CorA representation places the diplomatic string on
    # <token trans="...">. Preserve it and apply only the preregistered M19
    # alphabet normalization downstream.
    for e in root.iter():
        if e.tag.split('}')[-1]=='token':
            x=e.attrib.get('trans') or ''
            z=lib['norm_token'](x)
            if z:toks.append(z)
    if toks:return toks
    # Retain the original parser only as a fallback for alternate subcorpora.
    return lib['_parse_xml_original'](blob) if '_parse_xml_original' in lib else []

lib['_parse_xml_original']=lib['parse_xml']
lib['parse_xml']=parse_xml_token_trans
lib['main']()
