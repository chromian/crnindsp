# crnindsp

A Python toolbox for the identification of indicator species in a chemical reaction network.

## Overview

`crnindsp` is a Python module designed to analyze chemical reaction networks (CRNs) and identify indicator species—key species that provide insights into the multistability behavior of the network. It relies purely on two types of structural information:

- **Stoichiometry information**: The quantitative relationships between reactants and products in the network.
- **Qualitative regulatory information**: The regulatory interactions (e.g., activation or inhibition) within the network.

This toolbox is useful for researchers and scientists working in systems chemistry, computational biology, or related fields, particularly those studying multistable systems.

## Reference
The theoretical foundations of `crnindsp` are presented in:

[1] Huang, Yong-Jin, Atsushi Mochizuki, and Takashi Okada. "Identifying Phenotype-Indicative Molecules from the Structure of Biochemical Reaction Networks." _bioRxiv_ (2025): 2025-10.

## Installation

You can install `crnindsp` directly from GitHub using pip:

```bash
pip install git+https://github.com/chromian/crnindsp.git
```
