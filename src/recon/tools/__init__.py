"""Custom CrewAI tools for verification and research.

These tools implement Recon's fact-checking pipeline:
1. ClaimExtractorTool - Extract factual claims from markdown
2. CitationVerifierTool - Verify URLs contain claimed info
3. SemanticVerifierTool - LLM judges evidence-claim support
4. ConfidenceScorerTool - Score claim confidence (0.0-1.0)
5. SourceTrackerTool - Audit trail for provenance
6. ContradictionDetectorTool - Detect conflicting claims
"""

from recon.tools.citation_verifier import CitationVerifierTool
from recon.tools.claim_extractor import ClaimExtractorTool
from recon.tools.confidence_scorer import ConfidenceScorerTool
from recon.tools.contradiction_detector import ContradictionDetectorTool
from recon.tools.semantic_verifier import SemanticVerifierTool
from recon.tools.source_tracker import SourceTrackerTool

__all__ = [
    "ClaimExtractorTool",
    "CitationVerifierTool",
    "SemanticVerifierTool",
    "ConfidenceScorerTool",
    "ContradictionDetectorTool",
    "SourceTrackerTool",
]
