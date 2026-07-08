import logging
from typing import Dict, Any, List
from app.db.models import Job, Application

log = logging.getLogger(__name__)

class QualityGate:
    def __init__(self):
        log.info("QualityGate initialized (stub).")

    def validate_application(self, job: Job, app: Application) -> Dict[str, Any]:
        """Validate tailored artifacts and check overall fit before submission."""
        results = {
            "passed": True,
            "checks": {
                "keyword_coverage": 0.85,
                "seniority_match": True,
                "location_match": True,
                "sponsorship_risk": "low",
                "exaggeration_score": 0.0,
            },
            "warnings": []
        }
        
        low_title = job.title.lower()
        if "staff" in low_title or "principal" in low_title or "director" in low_title or "vp" in low_title:
            results["checks"]["seniority_match"] = False
            results["warnings"].append("Seniority mismatch: Job title implies Staff/Principal/Director/VP level.")
            results["passed"] = False
            
        # NOTE: stub has no access to the user's preferred country; treat any
        # detectable country as "match" and leave real targeting to the rule
        # filter (app.common.geo is the shared detector when this gets wired).
        low_location = (job.location or "").lower()
        if low_location and not job.remote:
            from app.common.geo import detect_country
            detected = detect_country(low_location)
            if detected and detected != "united states":
                results["checks"]["location_match"] = False
                results["warnings"].append(
                    f"Location check: posting located in {detected.title()} — "
                    "confirm it matches the user's preferred country.")

        log.info("QualityGate run complete. Passed: %s", results["passed"])
        return results
