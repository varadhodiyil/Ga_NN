#!/usr/bin/python
# -*- coding: latin-1 -*-

from rake_nltk import Rake , Metric

r = Rake(ranking_metric=Metric.WORD_FREQUENCY ,max_length=2 ) # Uses stopwords for english from NLTK, and all puntuation characters.

r.extract_keywords_from_text("""
Starting from the shell structure in atoms and the significant correlation
within electron pairs, we distinguish the exchange-correlation effects between
two electrons of opposite spins occupying the same orbital from the average
correlation among many electrons in a crystal. In the periodic potential of the
crystal with lattice constant larger than the effective Bohr radius of the
valence electrons, these correlated electron pairs can form a metastable energy
band above the corresponding single-electron band separated by an energy gap.
In order to determine if these metastable electron pairs can be stabilized, we
calculate the many-electron exchange-correlation renormalization and the
polaron correction to the two-band system with single electrons and electron
pairs. We find that the electron-phonon interaction is essential to
counterbalance the Coulomb repulsion and to stabilize the electron pairs. The
interplay of the electron-electron and electron-phonon interactions, manifested
in the exchange-correlation energies, polaron effects, and screening, is
responsible for the formation of electron pairs (bipolarons) that are located
on the Fermi surface of the single-electron band.
""" .encode("utf-8","ignore").__str__())

print(r.get_ranked_phrases()[:3]) # To get key