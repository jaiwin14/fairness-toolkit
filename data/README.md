# Data provenance

## compas-scores-two-years.csv
ProPublica COMPAS recidivism dataset. Included from the original repo.
Source: https://github.com/propublica/compas-analysis

## adult.data
UCI Adult / Census Income dataset (Becker & Kohavi, 1996), 32,561 rows,
original UCI training split. UCI's own archive (archive.ics.uci.edu) isn't
reachable from every network, so this was fetched from a mirror of the
identical, unmodified file hosted in the `shap` library's own data
repository (used by `shap.datasets.adult()`):

    https://github.com/shap/shap/raw/master/data/adult.data

Canonical source / full documentation:
https://archive.ics.uci.edu/dataset/2/adult

License: CC BY 4.0.

## german.data
UCI Statlog (German Credit Data) dataset (Hofmann, 1994), 1000 rows, 20
attributes. Same situation as adult.data: UCI's own archive isn't
reachable from every network. Fetched from an unmodified mirror in a
public fork of the AIF360 repository itself (aif360 ships instructions to
download this file but not the file itself):

    https://github.com/bhomass/AIF360/raw/master/aif360/data/raw/german/german.data

Canonical source / full documentation:
https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

License: CC BY 4.0.
