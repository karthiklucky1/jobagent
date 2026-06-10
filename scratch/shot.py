from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    pg = b.new_page(viewport={"width":1400,"height":1600})
    pg.goto("http://127.0.0.1:8000/dashboard", wait_until="networkidle")
    pg.wait_for_timeout(1500)
    pg.screenshot(path="scratch/dashboard.png", full_page=True)
    print("title:", pg.title())
    b.close()
