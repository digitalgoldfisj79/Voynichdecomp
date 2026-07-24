# Correction notice for notation packet v0.2

The v0.2 implementation used `m_core` in the absolute codelength calculation. `prefix + gallows + m_core + suffix` does not reconstruct 7,534 of 37,465 observed tokens. Absolute values presented as token codelengths were therefore codelengths of a lossy derived representation, not of the complete token stream.

The HMM-minus-IID sequence comparison remains a valid comparison on the common derived representation because the identical core term cancels. All absolute segmentation claims are superseded by v0.3's lossless two-part/prequential MDL analysis.
