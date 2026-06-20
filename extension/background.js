// HirePath Extension — Background Service Worker

// ── Session auth (token refresh) ─────────────────────────────────────────────
// The fill pack carries a short-lived Supabase access token plus (from the
// dashboard) a refresh token and Supabase credentials. We pull those secrets
// OUT of the pack and keep them only in the service worker's private storage —
// they are NEVER forwarded to the content script running on a third-party ATS
// page. The worker uses them to silently refresh the access token whenever an
// authed API call returns 401, so autofill keeps working through long,
// multi-step forms on any site instead of dying when the token expires.

function stashAuth(pack) {
  if (!pack || typeof pack !== "object") return;
  const auth = {};
  if (pack.refresh_token) auth.refresh_token = pack.refresh_token;
  if (pack.supabase_url) auth.supabase_url = pack.supabase_url;
  if (pack.supabase_anon_key) auth.supabase_anon_key = pack.supabase_anon_key;
  if (pack.auth_token) auth.access_token = pack.auth_token;
  // Strip the long-lived secrets so page content scripts can never read them.
  delete pack.refresh_token;
  delete pack.supabase_url;
  delete pack.supabase_anon_key;
  console.log(
    "[HirePath BG] stashAuth — refresh_token:", !!auth.refresh_token,
    "supabase_url:", !!auth.supabase_url, "access_token:", !!auth.access_token
  );
  if (!auth.refresh_token) {
    console.warn(
      "[HirePath BG] No refresh token in pack — the access token can't be renewed " +
      "when it expires. Ensure the HirePath dashboard is up to date and re-click Fill."
    );
  }
  if (!Object.keys(auth).length) return;
  // Merge over any previously stored creds (e.g. keep a rotated refresh token
  // if this pack didn't carry one).
  chrome.storage.local.get(["hirepath_auth"], (s) => {
    chrome.storage.local.set({ hirepath_auth: Object.assign({}, s.hirepath_auth || {}, auth) });
  });
}

async function doFetch(url, method, token, body) {
  const headers = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  try {
    const res = await fetch(url, {
      method: method || "POST",
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });
    let data = null;
    try { data = await res.json(); } catch (e) {}
    return { ok: res.ok, status: res.status, data };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

async function refreshAccessToken(auth) {
  // Exchange the refresh token for a new access token via Supabase's auth API.
  // Supabase rotates the refresh token on each call, so persist the new one.
  if (!auth.refresh_token || !auth.supabase_url || !auth.supabase_anon_key) return null;
  try {
    const res = await fetch(`${auth.supabase_url}/auth/v1/token?grant_type=refresh_token`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "apikey": auth.supabase_anon_key },
      body: JSON.stringify({ refresh_token: auth.refresh_token }),
    });
    if (!res.ok) return null;
    const data = await res.json();
    if (data && data.refresh_token) auth.refresh_token = data.refresh_token;
    return (data && data.access_token) || null;
  } catch (e) {
    return null;
  }
}

async function handleApiFetch(payload) {
  const store = await chrome.storage.local.get(["hirepath_auth"]);
  const auth = store.hirepath_auth || {};
  // Prefer the freshest access token the worker holds over the (possibly stale)
  // one the content script sent from its cached pack.
  const token = auth.access_token || payload.token;
  let result = await doFetch(payload.url, payload.method, token, payload.body);

  // Only attempt a refresh for calls that were actually authenticated.
  if (result.status === 401 && payload.token) {
    if (!auth.refresh_token || !auth.supabase_url || !auth.supabase_anon_key) {
      console.warn("[HirePath BG] 401 but no refresh creds available — cannot renew token");
    } else {
      const newToken = await refreshAccessToken(auth);
      if (newToken) {
        auth.access_token = newToken;
        await chrome.storage.local.set({ hirepath_auth: auth });
        console.log("[HirePath BG] Access token refreshed — retrying request");
        result = await doFetch(payload.url, payload.method, newToken, payload.body);
      } else {
        console.warn("[HirePath BG] Token refresh failed (refresh token rejected/expired)");
      }
    }
  }
  return result;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {

  // Popup sends FILL_JOB → send DO_FILL to the currently active tab
  if (msg.type === "FILL_JOB") {
    console.log("[HirePath BG] FILL_JOB received from popup");
    stashAuth(msg.payload);
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
    stashAuth(pack);
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
    handleApiFetch(msg.payload || {}).then(sendResponse);
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
