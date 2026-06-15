import asyncio
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.discovery.sources.linkedin_rapidapi import LinkedInRapidAPISource
from app.config import settings

async def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    
    print("Settings values:")
    print("  linkedin_rapidapi_enabled:", settings.linkedin_rapidapi_enabled)
    print("  rapidapi_key:", bool(settings.rapidapi_key))
    
    if not settings.rapidapi_key:
        print("Error: rapidapi_key is empty in settings! Please check .env file.")
        return
        
    source = LinkedInRapidAPISource()
    print("\nStarting LinkedIn RapidAPI fetch_jobs()...")
    jobs = await source.fetch_jobs()
    print(f"\nFetch complete! Retrieved {len(jobs)} jobs.")
    for idx, job in enumerate(jobs[:5]):
        print(f"  {idx+1}. {job.title} @ {job.company} | Location: {job.location} | URL: {job.url}")

if __name__ == "__main__":
    asyncio.run(main())
