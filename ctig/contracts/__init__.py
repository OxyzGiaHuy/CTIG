"""Automatic, evidence-grounded visual-contract construction."""

from .pipeline import ContractExtractionPipeline
from .wikipedia import BilingualWikipediaRetriever, WikipediaPassage

__all__ = ["BilingualWikipediaRetriever", "ContractExtractionPipeline", "WikipediaPassage"]
