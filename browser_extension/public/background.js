// public/background.js

const NATIVE_HOST_NAME = "com.safetypin.native";
let nativePort = null;
let isConnected = false;
const tabAnalysisState = new Map();
const ANALYSIS_COOLDOWN_MS = 8000;

function connectToNative() {
  console.log("Connecting to native host:", NATIVE_HOST_NAME);

  try {
    nativePort = chrome.runtime.connectNative(NATIVE_HOST_NAME);

    nativePort.onMessage.addListener(handleNativeMessage);
    nativePort.onDisconnect.addListener(handleNativeDisconnect);

    isConnected = true;
    console.log("✅ Connected to native host");

    chrome.action.setIcon({ path: "icons/icon-connected.png" });
    chrome.action.setBadgeText({ text: "✓" });
    chrome.action.setBadgeBackgroundColor({ color: "#4CAF50" });

    sendToNative({ type: "heartbeat" });

    return { success: true };
  } catch (error) {
    console.error("❌ Failed to connect to native host:", error);
    isConnected = false;

    chrome.action.setIcon({ path: "icons/icon-disconnected.png" });
    chrome.action.setBadgeText({ text: "✗" });
    chrome.action.setBadgeBackgroundColor({ color: "#f44336" });

    return { success: false, error: error.message };
  }
}

function handleNativeMessage(message) {
  console.log("Received from native:", message);

  // Forward message to any listening tabs/popups
  chrome.runtime
    .sendMessage({
      type: "nativeMessage",
      data: message,
    })
    .catch(() => {
      // No listeners, ignore
    });
}

function handleNativeDisconnect() {
  console.log("Native host disconnected");
  isConnected = false;
  nativePort = null;

  chrome.action.setIcon({ path: "icons/icon-disconnected.png" });
  chrome.action.setBadgeText({ text: "✗" });
  chrome.action.setBadgeBackgroundColor({ color: "#f44336" });

  // Notify listeners
  chrome.runtime
    .sendMessage({
      type: "nativeDisconnected",
    })
    .catch(() => {});
}

function sendToNative(message) {
  return new Promise((resolve) => {
    if (!nativePort || !isConnected) {
      resolve({ success: false, error: "Not connected" });
      return;
    }

    const messageId = Date.now() + Math.random();
    let responseHandler = null;
    const timeoutId = setTimeout(() => {
      if (nativePort && responseHandler) {
        nativePort.onMessage.removeListener(responseHandler);
      }
      resolve({ success: false, error: "Native host timed out" });
    }, 30000);

    responseHandler = (response) => {
      if (response.messageId === messageId) {
        clearTimeout(timeoutId);
        nativePort.onMessage.removeListener(responseHandler);
        resolve(response);
      }
    };

    nativePort.onMessage.addListener(responseHandler);
    nativePort.postMessage({ ...message, requestId: messageId, messageId });
  });
}

async function analyzePageDom(tabId) {
  if (!tabId) {
    return { success: false, error: "No tab ID" };
  }
  
  try {
    let response;

    // Try to message existing content script first (manifest already injects it)
    try {
      response = await chrome.tabs.sendMessage(tabId, {
        type: "analyze_page_dom"
      });
    } catch (e) {
      // If message failed, inject and retry once
      await chrome.scripting.executeScript({
        target: { tabId },
        files: ["content.js"]
      });

      response = await chrome.tabs.sendMessage(tabId, {
        type: "analyze_page_dom"
      });
    }
    
    if (!response?.domData) {
      return { success: false, error: "Failed to get DOM data" };
    }
    
    const domData = response.domData;
    console.log("Got DOM data from page:", { images: domData.images?.length, textElements: domData.textElements?.length });
    
    // Send to native host for analysis
    if (!nativePort || !isConnected) {
      return { success: false, error: "Not connected to native host" };
    }
    
    const messageId = Date.now() + Math.random();
    const nativeResponse = await new Promise((resolve, reject) => {
      let responseHandler = null;
      const timeoutId = setTimeout(() => {
        if (nativePort && responseHandler) {
          nativePort.onMessage.removeListener(responseHandler);
        }
        reject(new Error("Native host timed out"));
      }, 60000);
      
      responseHandler = (resp) => {
        if (resp.messageId === messageId) {
          clearTimeout(timeoutId);
          nativePort.onMessage.removeListener(responseHandler);
          resolve(resp);
        }
      };
      
      nativePort.onMessage.addListener(responseHandler);
      nativePort.postMessage({
        type: "dom_analysis",
        requestId: messageId,
        messageId: messageId,
        url: domData.url,
        html: domData.html,
        images: domData.images,
        textElements: domData.textElements
      });
    });
    
    console.log("Native host response:", nativeResponse);
    return { success: true, result: nativeResponse };
    
  } catch (error) {
    console.error("Error analyzing DOM:", error);
    return { success: false, error: error.message };
  }
}

// Listen for messages from React app
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log("Background received message:", message);

  switch (message.type) {
    case "connectNative":
      const result = connectToNative();
      sendResponse(result);
      break;

    case "disconnectNative":
      if (nativePort) {
        nativePort.disconnect();
        nativePort = null;
        isConnected = false;
      }
      sendResponse({ success: true });
      break;

    case "sendToNative":
      sendToNative(message.data).then(sendResponse);
      return true;

    case "analyzeDom":
      // First get DOM from content script, then send to native host
      analyzePageDom(message.tabId).then(sendResponse);
      return true;

    case "getConnectionStatus":
      sendResponse({ isConnected });
      break;

    default:
      sendResponse({ success: false, error: "Unknown message type" });
  }

  return true; // Keep channel open for async response
});

// Auto-connect on startup
chrome.runtime.onStartup.addListener(() => {
  connectToNative();
});

// Connect when extension is installed/updated
chrome.runtime.onInstalled.addListener(() => {
  connectToNative();
});

// Store for bad URLs
const badUrlCache = new Map();

// Check if URL is known bad from cache
function isKnownBadUrl(url) {
  return badUrlCache.has(url);
}

// Add URL to bad cache (in-memory only, clears on reload)
function blockBadUrl(url, reason) {
  badUrlCache.set(url, { reason, timestamp: Date.now() });
  console.log("Marked bad URL in cache:", url);
}

function hashCode(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i);
    hash |= 0;
  }
  return hash;
}

function isAnalyzableUrl(url) {
  return !!url &&
    !url.startsWith('chrome://') &&
    !url.startsWith('chrome-extension://') &&
    (url.startsWith('http://') || url.startsWith('https://'));
}

function scheduleAutoAnalysis(tabId, url, trigger, force = false) {
  if (!isAnalyzableUrl(url)) {
    return;
  }

  const now = Date.now();
  const state = tabAnalysisState.get(tabId) || {};

  if (state.inProgress) {
    console.log("Skipping analysis - already running for tab:", tabId, "trigger:", trigger);
    return;
  }

  if (!force && state.lastUrl === url && now - (state.lastRunAt || 0) < ANALYSIS_COOLDOWN_MS) {
    console.log("Skipping analysis - recently analyzed:", url, "trigger:", trigger);
    return;
  }

  console.log(`Scheduling analysis (${trigger}, force=${force}) for:`, url);
  tabAnalysisState.set(tabId, { inProgress: true, lastUrl: url, lastRunAt: now });

  setTimeout(() => {
    console.log("Starting auto-analysis for:", url, "trigger:", trigger);
    analyzePageDom(tabId).then(result => {
      console.log("Auto-analysis result:", result);
      console.log("Page classification:", result?.result?.page_classification);
      console.log("Action taken:", result?.result?.action_taken);
      const badImageElements = result?.result?.image_analysis?.bad_image_elements || [];
      const badTextElements = result?.result?.text_analysis?.bad_text_elements || [];
      const hasActionableBadElements = badImageElements.length > 0 || badTextElements.length > 0;

      if (result?.result?.page_classification === 'bad' || hasActionableBadElements) {
        var actionTaken = result.result.action_taken || 'page_blocked';

        blockBadUrl(url, result.result.page_reasons || []);

        if (actionTaken === 'page_blocked') {
          console.log("HIGH RISK DOMAIN - Blocking entire page");
          chrome.tabs.update(tabId, { url: 'blocked.html?url=' + encodeURIComponent(url) });
        } else {
          console.log("HIGH REPUTATION DOMAIN - Removing bad content only");
          chrome.tabs.sendMessage(tabId, {
            type: "remove_all_bad",
            badImages: badImageElements,
            badText: badTextElements,
              imageAction: "remove"
          }).catch(err => {
            console.log("Content removal error:", err.message);
          });
        }
      }
    }).catch(err => {
      console.log("Auto-analysis error:", err.message);
    }).finally(() => {
      const currentState = tabAnalysisState.get(tabId) || {};
      tabAnalysisState.set(tabId, {
        ...currentState,
        inProgress: false,
        lastRunAt: Date.now(),
        lastUrl: url
      });
    });
  }, 500);
}

// Auto-analyze pages as they load
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'loading') {
    const state = tabAnalysisState.get(tabId) || {};
    tabAnalysisState.set(tabId, {
      ...state,
      pendingLoadCycle: true
    });
    return;
  }

  if (changeInfo.status === 'complete' && tab.url) {
    const state = tabAnalysisState.get(tabId) || {};
    const forceReloadAnalysis = !!state.pendingLoadCycle;

    scheduleAutoAnalysis(tabId, tab.url, 'tabs.onUpdated.complete', forceReloadAnalysis);

    const nextState = tabAnalysisState.get(tabId) || {};
    tabAnalysisState.set(tabId, {
      ...nextState,
      pendingLoadCycle: false
    });
  }
});

chrome.webNavigation.onHistoryStateUpdated.addListener((details) => {
  scheduleAutoAnalysis(details.tabId, details.url, 'webNavigation.historyStateUpdated');
});

chrome.webNavigation.onReferenceFragmentUpdated.addListener((details) => {
  scheduleAutoAnalysis(details.tabId, details.url, 'webNavigation.referenceFragmentUpdated');
});

chrome.tabs.onRemoved.addListener((tabId) => {
  tabAnalysisState.delete(tabId);
});
