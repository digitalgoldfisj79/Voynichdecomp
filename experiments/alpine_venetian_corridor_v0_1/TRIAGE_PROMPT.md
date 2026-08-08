# Blind illustration triage prompt v0.1

Use this prompt identically for corridor and control pages. The model receives only an opaque page ID and image.

---

You are coding a medieval manuscript page for visual content. Do not identify the manuscript, author, artist, school, city, country, language, date, or holding institution. Do not compare the image with the Voynich Manuscript or any named manuscript. Do not infer provenance.

Return JSON only.

Allowed classes:

- plant
- root
- flower
- zodiac
- star_astronomy
- bath_human
- architecture_cartography
- diagram_geometry
- other_relevant
- none

For every visible relevant depiction, return a tight normalized bounding box `[x0,y0,x1,y1]` with coordinates in `[0,1]`, its class, confidence in `[0,1]`, and a purely visual description.

The description must emphasize observable structure rather than identity. Examples of permitted observations:

- branching count and topology;
- leaf placement and orientation;
- root/tuber/strand geometry;
- flower radial structure;
- human figure count and posture;
- tubs/vessels and their geometry;
- concentric rings, spokes, compartments and connectors;
- animal/sign pose without naming a zodiac tradition unless the sign is unambiguous;
- towers, roofs, crenellations, flags, walls and enclosure topology;
- star glyph ray count and arrangement.

Do not transcribe text except to record `text_present: true|false`. Do not use text to identify the manuscript.

Schema:

```json
{
  "opaque_page_id": "...",
  "page_relevant": true,
  "objects": [
    {
      "class": "plant",
      "bbox_norm": [0.0, 0.0, 1.0, 1.0],
      "confidence": 0.0,
      "description": "observable morphology only",
      "text_present": false
    }
  ],
  "page_notes": "visual/QA notes only"
}
```

If there is no relevant depiction, return `page_relevant=false`, `objects=[]`.

---

Any response naming a manuscript/place/school or explicitly comparing to the VMS is marked `identity_leak` and excluded from the blinded description arm.
