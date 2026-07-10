import sys
import os
import asyncio

# Add the project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.init_db import init_db
from app.discovery.registry import seed_registry, run_validation_loop

async def main():
    print("Step 1: Initializing database tables (including CompanyRegistry)...")
    init_db()
    
    print("\nStep 2: Seeding company registry from .env and bootstrap list...")
    new_entries = seed_registry()
    print(f"Added {new_entries} new boards to the registry.")

    if "--open-datasets" in sys.argv:
        from app.discovery.registry import seed_registry_from_open_datasets
        print("\nStep 2b: Bulk-seeding from the open ats-scrapers slug dataset (~20K companies)...")
        bulk = seed_registry_from_open_datasets()
        print(f"Added {bulk} boards from the open dataset.")
    
    print("\nStep 3: Running initial validation loop to activate seeded boards...")
    # Validate up to 100 entries immediately to bootstrap the pipeline
    validated = await run_validation_loop(limit=100)
    print(f"Validated {validated} boards in this run.")
    print("Registry seeding and validation completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
