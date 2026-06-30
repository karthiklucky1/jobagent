# HirePath — Project Explainer Video Guide

A complete, ready-to-record script and plan for explaining **HirePath** (AI Job‑Application
Copilot) on video. It covers what to prepare, what to say (word‑for‑word talking points), what
to show on screen, and how to open and close. Aimed at a **10–14 minute** walkthrough — trim the
optional sections for a 5‑minute version.

> Tip: Don't read this like a robot. These are *talking points*, not a teleprompter. Say them in
> your own words. The bold **"Say:"** lines are the spine; the *(Show: …)* lines tell you what to
> have on screen.

---

## 0. Before you record (preparation)

**Decide the video's goal first.** Pick ONE — it changes the tone:
- **Portfolio / recruiter demo** (most likely): show you can design a real system. Emphasize
  architecture decisions and the matching cascade.
- **Product demo for users:** emphasize the workflow and the "you always click Submit" safety.
- **Technical deep‑dive for engineers:** emphasize the pipeline, cost control, and grounding.

**Gear / software (free options):**
- **Screen recording:** OBS Studio (free), Loom, or the built‑in recorder (Windows Game Bar / Mac
  Shift‑Cmd‑5).
- **Audio:** a phone earbud mic beats a laptop mic. Record in a quiet, soft room (curtains, not bare
  walls) to kill echo.
- **Webcam bubble (optional):** a small circle in the corner builds trust. Not required.
- **Editing:** CapCut (free), DaVinci Resolve (free), or Clipchamp.

**Set up the app so the demo is smooth (do this BEFORE recording):**
```bash
# 1. Make sure it runs and has data already
python -m app.db.init_db
python -m scripts.run_discovery      # pull some jobs in advance
python -m scripts.run_matching       # so scores already exist on screen
python -m app.main                   # or: uvicorn app.api.server:app --port 8000
```
- Pre‑seed jobs so the dashboard isn't empty when you hit record.
- Have these tabs open and ready: landing page `/`, dashboard `/dashboard`, `/extension`,
  and your code editor on `app/matching/pipeline.py`.
- Zoom your editor font to ~16–18px so text is readable on small screens.
- Close personal tabs / notifications. Hide API keys — **never show `.env` on camera.**
- Do ONE practice run end‑to‑end without recording. It removes 90% of stumbles.

**Recording settings:** 1080p, 30fps is plenty. Record system audio + mic on separate tracks if you
can, so you can fix volume later.

---

## 1. The structure (your map)

| # | Section | Time | Purpose |
|---|---------|------|---------|
| 1 | Hook + who you are | 0:00–0:30 | Grab attention, say what this is |
| 2 | The problem | 0:30–1:30 | Why this project exists |
| 3 | What HirePath does (the 6 steps) | 1:30–3:00 | The big picture |
| 4 | Live demo: the workflow | 3:00–6:30 | Show, don't tell |
| 5 | Under the hood: matching cascade | 6:30–9:00 | Your engineering depth |
| 6 | Tailoring + grounding (anti‑hallucination) | 9:00–10:30 | The "trust" feature |
| 7 | Architecture & tech stack | 10:30–12:00 | How it's built |
| 8 | Safety, ethics & compliance | 12:00–12:45 | Maturity / judgment |
| 9 | Wrap‑up + call to action | 12:45–13:30 | What to do next |

---

## 2. Full script — section by section

### Section 1 — Hook (0:00–0:30)
*(Show: your face or the HirePath landing page `/`.)*

> **Say:** "Hi, I'm **[your name]**. Applying to jobs is exhausting — you scroll through hundreds of
> listings, most aren't a real fit, and you rewrite your résumé every single time. So I built
> **HirePath**, an AI job‑application copilot. It finds real, fresh tech roles, scores how well each
> one fits *your* résumé, drafts a tailored résumé and cover letter, and even fills out the
> application form for you — but **you** always click Submit. Let me show you how it works."

**Why this works:** problem + solution + the one memorable promise ("you always click Submit") in 30
seconds. Don't start with "So, um, this is a project I made…" — start with the pain.

---

### Section 2 — The problem (0:30–1:30)
*(Show: landing page, or a simple slide with 3 bullet points.)*

> **Say:** "Job hunting has three real problems. **One — discovery is noisy.** Job boards are full of
> reposts, ghost jobs that aren't really hiring, and roles that don't match your level. **Two —
> relevance is hard.** A keyword search can't tell that a 'Backend Engineer' role actually wants the
> exact stack on your résumé. **Three — tailoring takes forever.** Customizing a résumé and cover
> letter for each role is slow, and if you let an AI do it blindly, it *invents* experience you don't
> have, which gets you rejected or worse."
>
> "HirePath attacks all three: smart **discovery**, a multi‑stage **matching** system, and **grounded**
> tailoring that never makes things up."

---

### Section 3 — What it does, the 6 steps (1:30–3:00)
*(Show: the pipeline diagram. Read it left to right.)*

```
Discover ─▶ Match cascade ─▶ Score & enrich ─▶ Tailor (grounding check) ─▶ Auto-fill ─▶ You review & Submit
```

> **Say:** "At a high level, HirePath runs a pipeline. **Step one, Discover** — it pulls jobs directly
> from company applicant‑tracking systems like Greenhouse, Lever, Ashby, and Workday, plus public job
> feeds like RemoteOK, Remotive, YC, and Hacker News 'Who is hiring'. These are *public* sources, so
> it's compliant and the links are real."
>
> "**Step two, Match** — every job goes through a cascade of filters, cheapest first, expensive AI
> last, so only genuinely relevant roles reach me. I'll break that down in a minute because it's the
> heart of the project."
>
> "**Step three, Score and enrich** — each role gets a 0‑to‑100 fit score, a second‑opinion 'senior
> reviewer' verdict, ghost‑posting detection, a visa‑sponsorship check, and live hiring signals like
> how fresh the posting is."
>
> "**Step four, Tailor** — it drafts an ATS‑friendly résumé and cover letter, then runs a **grounding
> check** so nothing is invented beyond my real experience."
>
> "**Step five, Auto‑fill** — a browser extension fills the application form for me. And **step six,
> I review and submit.** The machine prepares, the human decides."

---

### Section 4 — Live demo (3:00–6:30) ⭐ *the most important part*
*(Show: the actual app. This is where people decide if your project is real.)*

Walk through it in this order. Narrate what you click.

**a) The dashboard / job board**
> **Say:** "This is the dashboard. Each card is a discovered job, ranked by priority. You can see the
> **fit score**, the reasoning behind it, the company, and signals like 'Posted 3 days ago'. There are
> views for the **Pipeline**, **All Jobs**, and **Boards**, plus a fit‑score distribution chart."

*(Show: hover over a high‑scoring job. Open it.)*

**b) A single job's detail**
> **Say:** "When I open a role, I see *why* it scored the way it did — the AI's reasoning, the senior
> reviewer's independent take, sponsorship assessment, and hiring‑intent signals. So I'm not guessing;
> I know which applications are worth my time."

**c) Tailoring**
> **Say:** "From here I can generate a tailored résumé and cover letter for this specific role. Notice
> it pulls real keywords from the job description but stays grounded in my actual résumé — it won't
> claim I know a technology I never listed."

**d) Auto‑fill via the extension**
*(Show: `/extension` page, then the form being filled — or the `gh_form.png` screenshot if a live
fill is risky on camera.)*
> **Say:** "The Chrome extension fills the application form right in my own browser, so I keep my login
> session and full control. It types with human‑like pacing. When it's done, **I** read it over and
> click Submit. HirePath never submits for me."

**e) The Telegram review loop (optional)**
> **Say:** "If I'm away from my desk, there's a Telegram bot that pings me to approve a job, answer a
> custom application question, or solve a CAPTCHA — so the review step works from my phone."

> **Demo safety tip:** if a live auto‑fill might glitch on camera, pre‑record it separately and cut it
> in, or show the screenshot. Never gamble the whole video on one live action.

---

### Section 5 — Under the hood: the matching cascade (6:30–9:00) ⭐ *your engineering story*
*(Show: `app/matching/pipeline.py` in the editor, or a slide of the table below.)*

> **Say:** "Here's the part I'm most proud of. Naively, you'd just ask an LLM to score every job — but
> that's slow and expensive at scale. So I built a **cascade**: cheap filters run first, and a job only
> reaches the expensive AI step if it survives every earlier gate."

Walk through each stage:

| Stage | What it does | Why it's here |
|-------|--------------|---------------|
| **0. Retrieval** | BM25 + FAISS embeddings (`all-MiniLM-L6-v2`) pull the top‑K candidate jobs | Narrow thousands of jobs to a handful, fast and free |
| **1. Rule filter** | Hard gates: title/seniority, location, job‑type, per‑company cap | Drop obvious non‑matches (e.g. internships) before spending anything |
| **2. Ghost filter** | Flags inactive or fake postings | Don't waste effort on jobs nobody's really hiring for |
| **3. Embedding gate** | Drops jobs below a cosine‑similarity floor | Cheap semantic relevance check |
| **4. LLM reranker** | Claude scores fit 0–100 with reasoning | The expensive, accurate step — only on survivors |
| **5. Hire probability** | Blends fit with hiring‑intent signals into a priority score | Rank by *likelihood of getting hired*, not just fit |
| **6. Senior review** | An independent 'senior engineer' AI verdict + score | A second opinion to catch the reranker's mistakes |

> **Say:** "The whole point is **cost control**. By the time I call the LLM in stage 4, I've already
> thrown out 95% of the noise with math that costs nothing. That's the difference between a toy and
> something that can run for many users every few hours."

> **Say (one decision to highlight):** "One design choice I like: stage 5 doesn't just rank by fit —
> it blends in *hiring intent*. A perfect‑fit job at a company that isn't really hiring is worth less
> than a good‑fit job that's clearly urgent. That's the hire‑probability score."

---

### Section 6 — Tailoring + grounding (9:00–10:30)
*(Show: `app/tailoring/` — mention `grounding.py`.)*

> **Say:** "The biggest risk with AI‑written résumés is **hallucination** — the model confidently adds
> skills or jobs you never had. That's an instant rejection, and it's dishonest. So tailoring runs a
> **grounding check**: every claim in the generated résumé and cover letter is verified against my real
> résumé. If it can't be supported by my actual experience, it gets cut. The AI rewrites and
> emphasizes what's true — it never invents."

> **Say:** "It also pulls ATS keywords from the job description so the résumé passes automated
> screening, but only keywords I can honestly back up."

---

### Section 7 — Architecture & tech stack (10:30–12:00)
*(Show: the project layout / a stack slide.)*

> **Say:** "On the tech side: the backend is **Python with FastAPI**. Data lives in **Supabase
> Postgres** in production, with a SQLite fallback for local dev, all through SQLModel. Auth is
> Supabase with JWTs verified on the server, and it's **multi‑tenant** — every query is scoped to the
> logged‑in user so no one ever sees anyone else's jobs or résumé."

> **Say:** "For matching I use **sentence‑transformers, FAISS, and BM25** for retrieval, and **Claude**
> as the main LLM for reranking, tailoring, and grounding. Automation is **Playwright plus a
> Manifest‑V3 Chrome extension**. There's a **Telegram bot** for the review loop, and **APScheduler**
> runs discovery and matching automatically about every six hours. The frontend is server‑rendered
> **Jinja templates with Tailwind and Chart.js** — no heavy JS framework."

> **Say (architecture note):** "It started as a single‑user personal agent and grew into a multi‑tenant
> web app — so you'll see that evolution in the code, like the single large `server.py` that holds the
> routes and dashboard."

---

### Section 8 — Safety, ethics & compliance (12:00–12:45)
*(Show: the README "Compliance & ethics" section, or a slide.)*

> **Say:** "I want to be clear about the ethics, because automation here can go wrong. **One — public
> sources only.** It uses public ATS APIs and feeds and respects robots.txt. **Two — no LinkedIn or
> Indeed automation;** their terms forbid it, so those are discovery‑only links I open myself. **Three —
> there's always a human review gate.** HirePath prepares and fills, but I approve and submit. And
> **four — grounded content**, so it never fabricates experience. The principle through the whole
> project is: **the machine prepares, the human decides.**"

---

### Section 9 — Wrap‑up + call to action (12:45–13:30)
*(Show: your face, or the landing page with the repo link.)*

> **Say:** "To recap: HirePath discovers real jobs, ranks them with a cost‑efficient matching cascade,
> tailors honest application materials, and fills the forms — while keeping me in control of every
> submission. It taught me a lot about pipeline design, controlling LLM cost, multi‑tenant
> architecture, and building AI you can actually trust."
>
> "The code and full write‑up are linked below / in the README. Thanks for watching — I'd love your
> feedback, and feel free to reach out."

**Optional CTA lines depending on goal:**
- Portfolio: "If you're hiring, I'd love to talk — my contact's in the description."
- Open source: "Star the repo if you found this interesting, and issues/PRs are welcome."

---

## 3. Quick 5‑minute version (if you need it short)

Keep only: Hook (Sec 1) → Problem in 2 sentences (Sec 2) → 6 steps fast (Sec 3) → Demo (Sec 4) →
Matching cascade in 60 seconds (Sec 5) → "you always click Submit" + CTA (Sec 9). Cut 6, 7, 8.

---

## 4. Delivery tips (so it sounds good)

- **Smile when you talk** — it changes your voice even on a screen recording.
- **Pause instead of "um."** Silence edits out cleanly; filler words don't.
- **One idea per sentence.** Short sentences sound confident.
- **Show the result first, explain second.** "Here's a tailored résumé — now let me show how it stays
  honest." People stay for payoff, then accept the detail.
- **Talk to one person**, not "you all / everybody." It feels personal.
- **Record in passes.** Do the demo in one take, the talking‑head intro/outro separately, and stitch
  them. Far less stressful than one perfect 14‑minute take.
- **Captions help a lot.** CapCut auto‑captions in a click; many people watch muted.

---

## 5. On‑screen checklist (paste this next to your monitor)

- [ ] App running with jobs already loaded
- [ ] `.env` / API keys NOT visible anywhere
- [ ] Editor font enlarged, on `app/matching/pipeline.py`
- [ ] Tabs ready: `/`, `/dashboard`, `/extension`, code editor
- [ ] Notifications off, personal tabs closed
- [ ] Mic tested, room quiet
- [ ] One practice run done
- [ ] Backup screenshot/clip of auto‑fill ready (`gh_form.png`)

---

*Generated as a planning aid. Adjust the timings and emphasis to match your audience and your own
voice.*
