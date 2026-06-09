"""
Factuality Guardrails Module (guardrails.py)
Implements deterministic security layers to prevent LLM hallucinations.
Verifies that all entities, numbers, dates, and articles mentioned in the LLM's response
exist directly in the retrieved source chunks.
"""

import re
from typing import List, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Guardrails")


class FactualityGuardrail:
    """
    Validates model responses for factual alignment with source documentation.
    Acts as a compliance gateway, rejecting answers containing unverified claims.
    """

    def __init__(self):
        # Regex to extract articles, sections, and numbers from text
        self.num_pattern = re.compile(r"\b\d+(?:\.\d+)?\b")
        self.article_pattern = re.compile(r"\b(?:article|art\.|section|chap\.|chapter)\s+\d+\b", re.IGNORECASE)

    def validate_response(
        self,
        llm_response: str,
        source_chunks: List[Dict[str, Any]],
        threshold_ratio: float = 1.0
    ) -> Tuple[bool, str, List[str]]:
        """
        Validates that the LLM response is strictly factual based on the source chunks.

        Checks:
        1. Entity/Numeric Check: All numbers, dates, and Article references in the response
           must be present in the source chunks.
        2. Citations Check: Ensures references are not fabricated.

        Returns:
            Tuple of:
            - is_valid: Boolean indicating if the response passed the check.
            - sanitized_response: The original response or a safe fallback string.
            - violations: List of strings detailing any found discrepancies.
        """
        logger.info("Running factuality guardrail checks on LLM response...")
        violations = []

        # 1. Consolidate all source text and metadata for validation
        source_text_unified = " ".join([
            f"{chunk['text']} {chunk['metadata'].get('hierarchy_path', '')} {chunk['metadata'].get('article', '')}"
            for chunk in source_chunks
        ]).lower()

        # 2. Extract and check specific Article/Section references
        response_references = self.article_pattern.findall(llm_response)
        for ref in response_references:
            # Clean spaces for comparison
            normalized_ref = re.sub(r"\s+", " ", ref).lower()
            if normalized_ref not in source_text_unified:
                violations.append(f"Unverified reference: '{ref}' was mentioned but not found in the retrieved context.")

        # 3. Extract and check all numeric values (dates, article numbers, fees)
        response_nums = set(self.num_pattern.findall(llm_response))
        source_nums = set(self.num_pattern.findall(source_text_unified))
        
        # Check if the response contains numbers not present in the sources
        # (e.g. LLM inventing 'September 2028' or '5% fee' when the source had '2% fee')
        unverified_nums = response_nums - source_nums
        
        # Filter out common small helper numbers to avoid false alarms (like 1, 2, 3 in bullet lists)
        unverified_nums = {num for num in unverified_nums if float(num) > 4.0}
        
        for num in unverified_nums:
            violations.append(f"Unverified number: '{num}' was mentioned but does not exist in the retrieved context.")

        # 4. Final Assessment
        if len(violations) > 0:
            logger.warning(f"Guardrail failed! Found {len(violations)} factuality violations.")
            # Fallback safe answer for compliance validation safety
            fallback_msg = (
                "Factuality Check Failure: The generated response contains details that could not be validated against the source circulars.\n"
                "Unverified claims:\n" + 
                "\n".join([f" - {v}" for v in violations]) + 
                "\n\nPlease review the search query or verify the uploaded PDF sources."
            )
            return False, fallback_msg, violations
        
        logger.info("Guardrail checks passed successfully.")
        return True, llm_response, []


if __name__ == "__main__":
    # Test stub
    guard = FactualityGuardrail()
    
    mock_sources = [
        {"text": "Under Article 5 of the EU AI Act, systems violating fundamental rights are banned. Deadline is 2026."}
    ]
    
    # Test valid response
    valid_text = "Article 5 states that AI systems violating rights are banned by 2026."
    ok, text, _ = guard.validate_response(valid_text, mock_sources)
    print("Test 1 (Valid) Pass:", ok)
    
    # Test invalid response (inventing Article 99 and year 2029)
    invalid_text = "Article 99 states that AI systems are banned by 2029."
    ok, text, violations = guard.validate_response(invalid_text, mock_sources)
    print("Test 2 (Invalid) Pass:", ok)
    print("Blocked Output:\n", text)
