// HirePath Extension — Background Service Worker

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {

  // Popup sends FILL_JOB → send DO_FILL to the currently active tab
  if (msg.type === "FILL_JOB") {
    console.log("[HirePath BG] FILL_JOB received from popup");
    chrome.storage.local.set({ hirepath_fill_pack: msg.payload, hirepath_auto_fill: false }, () => {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (!tabs[0]) { sendResponse({ ok: false, error: "No active tab" }); return; }
        console.log("[HirePath BG] Sending DO_FILL to tab", tabs[0].id, tabs[0].url);
        chrome.tabs.sendMessage(tabs[0].id, { type: "DO_FILL", fillPack: msg.payload }, (res) => {
          sendResponse(res || { ok: true });
        });
      });
    });
    return true;
  }

  // Content script (on dashboard page) sends OPEN_AND_FILL → open job tab, then fill when loaded
  if (msg.type === "OPEN_AND_FILL") {
    const pack = msg.payload;
    console.log("[HirePath BG] OPEN_AND_FILL received for:", pack?.job_title, pack?.apply_url);
    // Store BOTH a one-shot auto_fill flag AND a persistent copilot session.
    // The copilot session (10-min window) lets autofill survive cross-domain
    // navigations — e.g. accenture.com → myworkdayjobs.com after clicking Apply.
    chrome.storage.local.set({
      hirepath_fill_pack: pack,
      hirepath_auto_fill: true,
      hirepath_copilot_pack: pack,
      hirepath_copilot_ts: Date.now(),
    }, () => {
      chrome.tabs.create({ url: pack.apply_url }, (tab) => {
        console.log("[HirePath BG] Opened tab", tab.id, "for", pack.apply_url);
        sendResponse({ ok: true, tabId: tab.id });
      });
    });
    return true;
  }

  if (msg.type === "PING") {
    sendResponse({ ok: true, version: chrome.runtime.getManifest().version });
  }

  // Content script asks the background worker to make a cross-origin API call.
  // Content-script fetches run in the page origin and get blocked by CORS;
  // the service worker has host_permissions and is exempt.
  if (msg.type === "API_FETCH") {
    const { url, method, token, body } = msg.payload || {};
    const headers = { "Content-Type": "application/json" };
    if (token) headers["Authorization"] = `Bearer ${token}`;
    fetch(url, {
      method: method || "POST",
      headers,
      body: body ? JSON.stringify(body) : undefined,
    })
      .then(async (res) => {
        let data = null;
        try { data = await res.json(); } catch (e) {}
        sendResponse({ ok: res.ok, status: res.status, data });
      })
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true; // keep the message channel open for the async response
  }
});

const ATS_HOSTS = /greenhouse\.io|lever\.co|ashbyhq\.com|myworkdayjobs\.com|workday\.com|smartrecruiters\.com|avature\.net|icims\.com|taleo\.net|successfactors|brassring|jobvite\.com|workable\.com|bamboohr\.com|linkedin\.com|indeed\.com/i;

// When a tab finishes loading, check if we should auto-fill it
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status !== "complete") return;
  if (!tab.url || tab.url.startsWith("chrome")) return;

  chrome.storage.local.get(
    ["hirepath_fill_pack", "hirepath_auto_fill", "hirepath_copilot_pack", "hirepath_copilot_ts"],
    (data) => {
      const pack = data.hirepath_fill_pack || data.hirepath_copilot_pack;
      if (!pack) {
        console.log("[HirePath BG] Tab", tabId, "loaded but no pack in storage — skipping");
        return;
      }

      let tabHost;
      try { tabHost = new URL(tab.url).hostname; } catch (_) { return; }

      // Diagnostic dump
      console.log("[HirePath BG] === Tab loaded:", tabId, tabHost, "===");
      console.log("[HirePath BG]   auto_fill:", data.hirepath_auto_fill);
      console.log("[HirePath BG]   copilot_pack:", !!data.hirepath_copilot_pack);
      console.log("[HirePath BG]   copilot_ts:", data.hirepath_copilot_ts);
      console.log("[HirePath BG]   ATS match:", ATS_HOSTS.test(tabHost));

      // Determine if this tab should receive DO_FILL:
      // 1. One-shot flag set when we opened the tab (exact host match OR known ATS)
      // 2. Persistent copilot session active (30-min window) on any ATS/actionable page
      const SESSION_MS = 30 * 60 * 1000;
      const sessionAge = data.hirepath_copilot_ts ? (Date.now() - data.hirepath_copilot_ts) : Infinity;
      const freshSession = sessionAge < SESSION_MS;
      // Also treat having a copilot_pack on any known ATS as valid
      // (handles cases where timestamp was lost but pack data is still present)
      const hasPackOnATS = !!data.hirepath_copilot_pack && ATS_HOSTS.test(tabHost);

      let shouldFill = false;
      if (data.hirepath_auto_fill) {
        try {
          const jobHost = new URL(pack.apply_url || "").hostname;
          // Same host OR tab is a known ATS (handles accenture → workday cross-domain)
          if (tabHost === jobHost || ATS_HOSTS.test(tabHost)) shouldFill = true;
        } catch (_) {
          if (ATS_HOSTS.test(tabHost)) shouldFill = true;
        }
      } else if ((freshSession || hasPackOnATS) && ATS_HOSTS.test(tabHost)) {
        // Copilot session: resume on any ATS page automatically
        shouldFill = true;
      }

      console.log("[HirePath BG]   shouldFill:", shouldFill, "freshSession:", freshSession, "hasPackOnATS:", hasPackOnATS);

      if (!shouldFill) return;

      console.log("[HirePath BG] ▶ Tab", tabId, "matched — sending DO_FILL in 2s");
      if (data.hirepath_auto_fill) chrome.storage.local.set({ hirepath_auto_fill: false });

      setTimeout(() => {
        chrome.tabs.sendMessage(tabId, { type: "DO_FILL", fillPack: pack }, (res) => {
          if (chrome.runtime.lastError) {
            console.warn("[HirePath BG] Could not send DO_FILL:", chrome.runtime.lastError.message);
          } else {
            console.log("[HirePath BG] DO_FILL sent, response:", res);
          }
        });
      }, 2000);
    }
  );
});
