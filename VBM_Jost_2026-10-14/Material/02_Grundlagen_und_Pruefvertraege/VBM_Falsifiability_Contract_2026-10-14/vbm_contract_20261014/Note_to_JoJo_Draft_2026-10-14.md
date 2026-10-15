Hi JoJo,

I went through your v138 package independently, and I think the most useful next step is to separate two questions: whether a VBM-type cipher can be built, and whether the actual Voynich manuscript used it.

The code reproduces your published output, but v138 has some implementation problems. In particular, rare consonant clusters collide in the cell table, and a number of final tokens still contain glyph combinations absent from the supplied B reference. We built a different compositional cipher that decodes its own German input exactly, including the Hartlieb text, but its VMS surface fails; moreover, it places its bridges inside source-aligned words, so it doesn't validate your distinctive cross-space proposal.

I don't think another round of tuning hapaxes, entropy or prefixes will settle the historical question. Your key idea is stronger and more specific than any of those statistics: the two sides of EVA spaces jointly represent vowels, while internal material carries consonantal structure.

Could you help us freeze five points, as the *essential* VBM rather than the details of v138?

1. Must all ordinary EVA gaps carry a vowel bridge, or only a defined subset? What are the allowed exceptions?
2. Which bridge and consonant mappings stay constant, and what permits them to change between contexts or sections?
3. Is the original text meant to be recoverable exactly after normalisation, or is information deliberately lost?
4. Is REVA a necessary component, or one of two competing variants?
5. What possible observation would convince you that a fixed version of VBM is wrong?

That would allow us to identify a proper discriminator, ideally a text correspondence established independently of the proposed decipherment. Otherwise a slot generator can reproduce the same structural statistics without encoding any plaintext, and we cannot choose between them from those statistics alone.

This isn't an argument that VBM is wrong. It is an attempt to define the smallest claim that could genuinely be tested. I can share the audit and the working code if useful.

Ed
