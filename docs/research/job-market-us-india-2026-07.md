# The 2025–26 Job Market: USA & India — Data Report for SpotApply Strategy (July 2026)

**Method.** Deep-research run (107 agents: 5 search angles → 25 source fetches → claim
extraction → adversarial 3-vote verification), interrupted mid-verification by an API
session limit and completed manually from the run journal + targeted gap-fill searches.
**Confidence labels:** `[3-0]` = survived adversarial verification 3-0 ·
`[primary]` = extracted from the named primary source with the supporting quote
captured, adversarial pass incomplete · `[gap-fill]` = single-pass search synthesis,
treat as directional. Vendor self-reports are flagged. Compiled 2026-07-19.

---

## 1. USA — macro state of the tech job market

- Tech occupation employment grew by **47,000 workers in June 2026**; tech unemployment
  fell to **2.9% vs 4.2% national**. `[3-0]` (CompTIA Tech Jobs Report)
- **280,000+ new US tech job postings in June 2026**; June was the second consecutive
  month active tech postings topped **600,000** — the addressable posting volume for a
  US tech job tool. `[3-0]` (CompTIA)
- Net tech employment contracted ~0.3% in 2025 (≈33,600 fewer workers, to ~9.6M), but
  CompTIA projects **+1.9% in 2026** (~185K new jobs), with tech workforce growth
  projected at 2x the overall US workforce over the next decade. `[primary]` (CompTIA)
- The software-developer posting cycle, measured (Indeed/FRED index, Feb 2020 = 100):
  peak **233.86** (Feb 2022) → **72.69** (Jan 2024) → bottom **61.12 on May 17, 2025**
  → **74.56 on July 10, 2026** (+14% YoY). Software postings remain **~25% below
  pre-pandemic baseline** while overall US postings are back at baseline (101.17).
  `[primary]` (FRED series IHLIDXUSTPSOFTDEVE — mid-2025 was the trough; a real
  recovery is underway but incomplete)

## 2. USA — AI's measurable impact on roles

- US software-dev postings on Indeed rose **almost 15%** between Claude Code's launch
  (late Feb 2025) and mid-2026, while overall US postings **declined 7%**. `[3-0]`
  (Indeed Hiring Lab, July 2026 — vendor platform data but primary)
- The rebound is top-heavy: **71% of the net increase in software postings (May 2025 →
  May 2026) is senior roles; 37% mentions AI in the title** (overlapping). Demand
  growth is for experienced, AI-fluent engineers — not entry-level. `[3-0]` (Hiring Lab)
- The AI-exposure correlation **flipped sign**: most-AI-exposed occupations fell hardest
  2022–2026, but over the last year they've rebounded hardest — AI's posting effect is
  shifting from destruction toward creation. `[3-0]` (Hiring Lab)
- AI skills went from niche to default: share of tech postings citing ≥1 AI skill rose
  **15% (Jan 2024) → 73% (May 2026)**; 275K+ active postings referenced AI skills in
  Jan 2026; "AI engineer" postings +81% YoY. `[primary]` (CompTIA)
- Non-tech-sector employers implementing AI are driving 2026 tech hiring, offsetting
  tech-company layoffs (top AI-skill hiring sectors: tech, professional services,
  finance/insurance, manufacturing). `[primary]` (CompTIA)
- AI was the *cited* reason for just over **10,000 US job cuts in the first 7 months of
  2025** (Challenger, Gray & Christmas) — small against total churn; the entry-level
  effect (below) is the bigger structural story. `[primary]`

## 3. USA — the entry-level collapse (the defining seeker pain)

- New grads are **~7% of Big Tech hires** (was ~15% pre-pandemic), down ~25% from 2023
  and **>50% below 2019**; at VC-backed startups <6% of hires (down ~11% from 2023,
  >30% from 2019). `[primary]` (SignalFire State of Talent 2025 — VC self-published;
  corroborated by TechCrunch + Fortune coverage)
- Handshake entry-level corporate postings ~-15% YoY (2024); intern→full-time offer
  conversion fell below 51% in 2023-24, lowest in 5+ years (NACE). `[primary]` (Fortune)
- 37% of managers say they'd rather use AI than hire a Gen Z employee (third-party
  survey relayed by SignalFire — weakest provenance in this set). `[primary]`
- **Concrete example:** CS grad Kenneth Kang (Portland, OR) submitted **2,500+
  applications in his first year** out of college → 10 interviews → 10+ months to land
  a job (at Adidas, where he'd interned). `[primary]` (Fortune, Aug 2025)
- AI engineer geography is re-concentrating: SF + NYC now hold **65% of US AI
  engineers**. `[primary]` (SignalFire)

## 4. USA — seeker behavior & the application flood

- LinkedIn applications **+45% YoY (Oct 2025), ~9,500 submitted per minute**.
  `[primary]` (CNBC, LinkedIn representative)
- Recruiter reality: popular roles get **300–500 applications within 3 days**, 1,000+
  over a weekend; a recruiter estimates +25% AI-assisted applications in months before
  Oct 2025 and says volume has slowed hiring "down to a crawl." `[primary]` (CNBC)
- Recruiters say generic AI applications lose to tailored ones, and spray-applying is
  counterproductive — "spend the time of 7-8 applications on one exceptionally clear
  one." `[primary]` (CNBC — opinion, but consistent with Huntr's measured 9.25% vs
  2.58% interview-rate data in our July 10 research)

## 5. USA — recruiter/ATS landscape and what's publicly reachable

- ATS share of US tech/enterprise postings (2024, Ongig/Phenom via secondary source —
  page 403'd, recovered from search index; treat as approximate): **Workday ~32%,
  Greenhouse ~18%, Lever ~12%, iCIMS ~10%, Ashby ~5% (fastest-growing)**; top-8
  platforms ≈85% of enterprise/tech postings. `[primary-weak]`
- Segmentation: Ashby concentrates in high-growth startups; iCIMS in healthcare/retail/
  hospitality; Workday/iCIMS dominate enterprise & non-tech. SpotApply's current
  Greenhouse/Lever/Ashby depth maps to the startup/tech segment. `[primary]`
- Application friction varies ~5x by ATS: **Workday 22–35 min manually vs Lever 4–7
  min** — Workday is both the biggest coverage gap and the biggest autofill prize.
  `[primary]` (Jobaholic — a competing autofill vendor; self-interested but plausible)
- **Public scrapability is proven at scale:** an open-source aggregator scrapes
  Greenhouse, Lever, Ashby, BambooHR, iCIMS, Paylocity AND Workday public APIs without
  auth — **1M+ active postings from 20K+ companies, refreshed daily via GitHub Actions
  as of Jul 18, 2026**; ~95K company slugs harvested from Common Crawl CDX archives;
  per-platform concurrency tuned (Workday 50 workers, Ashby 5). `[primary]`
  (github.com/Feashliaa/job-board-aggregator — validates and extends our registry
  growth plan, incl. that Workday and iCIMS have publicly reachable endpoints)

## 6. India — macro state of the white-collar market

All JobSpeak numbers are **vendor self-reports** (Naukri's own platform index — but the
de facto primary indicator for Indian white-collar hiring). `[primary]`

- **Feb 2026: JobSpeak index 3,233, +12% YoY** — strongest February in years; Jan→Feb
  MoM +23% vs typical 13–16% seasonal range.
- Jan 2026: 2,637, +3% YoY. Dec 2025: 3,001, +13% YoY. `[gap-fill for Dec]`
- **IT hiring recovered**: +6% YoY overall in Feb 2026, IT fresher hiring +8% (after
  being flat YoY in Jan). AI/ML roles: **+34% YoY (Jan), +41% within IT (Feb)**; Indian
  MNCs grew AI/ML hiring **+82% YoY vs +43% at foreign MNCs**.
- **Non-IT anchors growth**: Feb 2026 — Insurance +28%, BPO/ITES +22%, Real Estate
  +19%, Hospitality +15%, Retail +14%. Jan 2026 — BPO/ITES +21%, Hospitality +15%;
  Banking/Financial Services **-15% YoY**.
- **Fresher demand on Naukri is strong**: 0–3 yr hiring **+17% YoY (Feb 2026)**, +8%
  (Jan); 20+ LPA premium roles +23%. Fresher growth is concentrated in non-IT: Real
  Estate +42%, BPO/ITES +39%, Insurance +35%, Hospitality +33%.

## 7. India — IT services vs GCCs (the structural shift)

- **TCS cut ~12,000 jobs (~2% of 613K workforce), confirmed Jul 28, 2025** —
  concentrated in mid-to-senior and middle management, not freshers; simultaneously
  raised pay for ~80% of remaining staff (skills repricing, not pure cost-cutting);
  CEO publicly denied AI-driven headcount reduction; 35-day bench policy now strictly
  enforced. `[primary]` (JobsPikr synthesis of announcements)
- **EY estimate: Indian IT services have cut entry-level (fresher) roles 20–25% due to
  automation/AI** — the traditional campus→IT-services pipeline is shrinking.
  `[primary]` (Rest of World)
- **GCCs are the counterweight**: forecast ~+50% hiring in FY26 per one source
  `[primary-weak]`; NASSCOM community data is more sober — GCC openings fell 8–12% QoQ
  in Q1 2025 (BFSI GCCs -10–15%), with FY26 recovery of up to +10% concentrated in
  GenAI, platform engineering, data security; GCC attrition of 12–15%/yr provides a
  steady backfill-posting baseline. `[primary]`
- **Campus example:** at IIITDM Jabalpur (a top-tier institute), **fewer than 25% of
  ~400 students in the class ending May 2026 had job offers**. `[primary]` (Rest of
  World) — reconcile with Naukri's +17% fresher demand: the growth is non-IT and
  off-campus; the elite-campus→IT-services escalator specifically is broken.
- Junior-role contraction is global: LinkedIn/Indeed/EURES recorded **-35% junior tech
  positions across major EU countries in 2024**. `[primary]`

## 8. India — where seekers and employers actually are `[gap-fill]`

- **Naukri dominates white-collar**: ~62–70% market share claims, 78M+ resume database,
  76K+ corporate clients (Info Edge, public co). LinkedIn India: **130M+ users** —
  one of LinkedIn's largest markets. foundit (ex-Monster): ~800K active jobs. Apna:
  fastest-growing for freshers/Tier-2-3 cities. Instahyre/Cutshort: curated tech
  niches. Practical takeaway repeated across sources: Naukri + one aggregator covers
  ~80% of white-collar opportunity.
- **India ATS landscape**: Zoho Recruit, Keka, Darwinbox, Freshteam dominate domestic
  ATS usage (India ATS market ~$0.30B 2024 → $0.50B 2033); startups → Zoho/Freshteam,
  enterprises → Darwinbox (+ Greenhouse for global-facing product cos). Most Indian
  hiring still flows through job portals (Naukri/LinkedIn postings + recruiter
  database search) rather than US-style public ATS career APIs — **the ATS-API
  discovery model that powers SpotApply's US pipeline does NOT transfer 1:1 to
  India**; Indian product companies & GCCs using Greenhouse/Lever/Ashby/Workday are
  the transferable slice.

## 9. Salaries — the benchmark gap `[gap-fill]`

| Role/segment | USA | India |
|---|---|---|
| Software dev, median | BLS ~$132K; Levels.fyi median TC ~$192K (P25 $135K / P75 $277K) | — |
| Fresher, IT services | — | ₹3–5 LPA (stagnant ~a decade) |
| Fresher, product cos | — | ₹6–12 LPA; AI/ML ₹7–15 LPA |
| Mid (3–5 yr), GCC Bangalore | — | ₹18–28 LPA fixed |
| Senior (6–9 yr), GCC | — | ₹30–45 LPA (total ₹35–65 LPA; Pune/Hyderabad 10–20% lower) |
| AI/ML premium | — | +25–40% over general SWE; demand/supply ≈3x |

Wage-level context for the US visa engine: a ₹-to-$ career jump plus the DHS weighted
lottery means the *offered US salary band* now drives visa odds — the Level 1 vs
Level 4 spread (below) is worth more than any resume tweak.

## 10. The India→USA pipeline — policy shock in progress

All USCIS/DHS claims from primary government pages. `[primary]`

- **$100K H-1B fee** (proclamation signed Sep 19, 2025): applies to any NEW H-1B
  petition after Sep 21, 2025, including the 2026 lottery. NOT retroactive; one-time;
  renewals/extensions unaffected; existing holders can travel freely; national-interest
  exceptions "extraordinarily rare."
- **Weighted lottery final rule effective Feb 27, 2026** (first applies FY2027 season):
  selection entries scale with OEWS wage level — **Level IV = 4 entries, III = 3,
  II = 2, Level I = 1**. Entry-level odds cut to ~¼ of top-band odds. DHS's stated
  intent: shift allocation to higher-paid roles; explicit pressure on the IT-services
  staffing model. FY2027 registration: **Mar 4–19, 2026**, selections by Mar 31.
- **Student pipeline is contracting hard**: new Indian student arrivals **-44.5% YoY in
  Aug 2025** (74,825 → 41,540) and -46.4% in Jul 2025 (ADIS/I-94 via ITA; Forbes) —
  combined Jul-Aug ≈ -50%. State Dept suspended student-visa interviews ~3 weeks
  (May–Jun 2025) for social-media vetting. USCIS director has said he hopes to END
  OPT; 54% of intl grad students say they wouldn't have attended a US university
  without OPT (IFP/NAFSA survey).
- **Net effect for SpotApply's visa segment**: the *stock* of current OPT/H-1B
  candidates in the US needs wage-level-aware targeting NOW (change-of-status filers
  avoid the $100K fee; job's wage level sets their odds) — while the *flow* of new
  students is shrinking, meaning this wedge segment peaks over the next ~2-3 years and
  the India-domestic market grows in relative importance.

---

## 11. Implications — ranked opportunities for SpotApply

1. **US senior+AI tech roles are the core market, and it's recovering.** 600K+ active
   tech postings, demand concentrated in senior/AI-fluent roles — exactly the profile
   that pays $10/mo without blinking. Double down on AI-skill surfacing (73% of
   postings now cite AI skills → make "AI-skill match" explicit in scoring/tailoring).
2. **Visa Odds Engine is urgent and time-boxed.** The FY2027 weighted-lottery window
   (registrations March 2026, now past; FY2028 next) + the $100K fee's
   change-of-status exemption make wage-level intelligence the single highest-value
   feature for the most desperate, highest-paying segment — but the student-flow data
   says this segment shrinks after ~2027. Ship it now, harvest it now.
3. **Registry expansion is validated and cheap.** A solo open-source project reaches
   1M+ postings/20K companies across 7 ATSes including *Workday* public endpoints,
   with Common-Crawl slug harvesting (~95K slugs) — SpotApply's registry (Greenhouse/
   Lever/Ashby native) should add Workday + iCIMS + Paylocity public endpoints and the
   Common-Crawl harvester. Workday alone ≈ ~32% of enterprise postings and its 22–35
   min manual forms make it the highest-value autofill target.
4. **The entry-level crisis is a seeker-pain goldmine but a monetization trap.** New
   grads face 2,500-application searches (US) and <25% campus placement (India elite
   campuses) — huge volume, low willingness-to-pay. Serve with the free tier +
   SpotCheck wedge + honest "aim, don't spray" coaching (recruiters confirm spray
   fails); convert the ones who land interviews.
5. **India expansion is real but requires a different product.** The market is growing
   (+12% YoY Feb 2026) and fresher demand is strong — but it's non-IT-led, flows
   through Naukri/LinkedIn rather than public ATS APIs, and fresher pay (₹3–5 LPA)
   can't support US pricing. The transferable beachhead: **GCCs + Indian product
   companies on global ATSes** (Greenhouse/Lever/Ashby/Workday) hiring at ₹18–65 LPA —
   discoverable with SpotApply's existing pipeline today. Naukri-based discovery would
   require aggregator/partner data (TheirStack lists a Naukri source) or compliance
   review — do NOT scrape Naukri directly without one.
6. **Two-sided leverage grows.** The application flood (9,500/min on LinkedIn; 300–500
   apps per role in 3 days) keeps strengthening the Recruiter Bridge thesis: recruiters
   are the ones asking for triage.

**Watch-list / caveats:** JobSpeak and all platform posting counts are vendor
self-reports; the ATS market-share table came from a 403'd page recovered via search
index; GCC "+50% FY26 hiring" conflicts with NASSCOM's more cautious 8–12% Q1 decline
and up-to-+10% recovery — trust NASSCOM's number for planning. The adversarial
verification pass completed for only 5 of 79 claims before the API session limit; the
20 claims queued for verification all carry captured quotes from their primary sources.

### Primary sources
Indeed Hiring Lab (Jul 2026) · CompTIA Tech Jobs Report (Jun–Jul 2026) · FRED
IHLIDXUSTPSOFTDEVE · Naukri JobSpeak blog (Jan/Feb 2026) · JobsPikr TCS/GCC analysis ·
NASSCOM GCC community (FY25/FY26) · SignalFire State of Talent 2025 · CNBC (Oct 2025) ·
Fortune (Aug 2025) · Rest of World (2025) · USCIS H-1B FAQ · USCIS/DHS weighted-lottery
release · Forbes/ITA ADIS-I-94 student data · gap-fill: bestjobsearchapps/Internshala
(India portals), IMARC (India ATS market), BLS/Levels.fyi/Talhive/PlugScale (salaries),
GitHub job-board-aggregator (ATS scrapability).
