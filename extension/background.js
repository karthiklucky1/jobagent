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

// When a tab finishes loading, check if we should auto-fill it
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status !== "complete") return;
  chrome.storage.local.get(["hirepath_fill_pack", "hirepath_auto_fill"], (data) => {
    if (!data.hirepath_auto_fill || !data.hirepath_fill_pack) return;
    const pack = data.hirepath_fill_pack;
    if (!tab.url || !pack.apply_url) return;
    try {
      const tabHost = new URL(tab.url).hostname;
      const jobHost = new URL(pack.apply_url).hostname;
      if (tabHost !== jobHost) return;
    } catch (e) { return; }

    console.log("[HirePath BG] Tab", tabId, "matches job host — sending DO_FILL in 2.5s");
    // Clear flag so re-navigations don't re-fire
    chrome.storage.local.set({ hirepath_auto_fill: false });
    setTimeout(() => {
      chrome.tabs.sendMessage(tabId, { type: "DO_FILL", fillPack: pack }, (res) => {
        if (chrome.runtime.lastError) {
          console.warn("[HirePath BG] Could not send DO_FILL to tab:", chrome.runtime.lastError.message);
        } else {
          console.log("[HirePath BG] DO_FILL sent, response:", res);
        }
      });
    }, 2500);
  });
});
