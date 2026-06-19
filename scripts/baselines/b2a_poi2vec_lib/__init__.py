"""B2a POI2Vec (Feng et al., AAAI 2017) — standalone per-POI 64-d substrate.

This is the BASELINE POI2Vec, NOT the in-repo fclass-level HGI teacher in
``research/embeddings/hgi/poi2vec.py``. See ``model.py`` docstring for the
faithfulness statement + deviations.
"""
from .model import GeoPOI2Vec, build_geo_binary_tree

__all__ = ["GeoPOI2Vec", "build_geo_binary_tree"]
