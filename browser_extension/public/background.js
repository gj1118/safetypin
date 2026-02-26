// public/background.js

const NATIVE_HOST_NAME = "com.safetypin.native";
let nativePort = null;
let isConnected = false;

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
    const timeoutId = setTimeout(() => {
      resolve({ success: false, error: "Native host timed out" });
    }, 30000);

    const responseHandler = (response) => {
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
    // First, inject content script if not already loaded
    try {
      await chrome.scripting.executeScript({
        target: { tabId },
        files: ['content.js']
      });
    } catch (e) {
      // Script might already be loaded
    }
    
    // Get DOM data from content script
    const response = await chrome.tabs.sendMessage(tabId, {
      type: "analyze_page_dom"
    });
    
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
      const timeoutId = setTimeout(() => {
        reject(new Error("Native host timed out"));
      }, 60000);
      
      const responseHandler = (resp) => {
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

// Auto-analyze pages as they load
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'loading' && tab.url && !tab.url.startsWith('chrome://') && !tab.url.startsWith('chrome-extension://')) {
    console.log("Tab loading:", tab.url);
    
    // Small delay to let DOM start building
    setTimeout(() => {
      console.log("Starting auto-analysis for:", tab.url);
      analyzePageDom(tabId).then(result => {
        console.log("Auto-analysis result:", result);
        console.log("Page classification:", result?.result?.page_classification);
        console.log("Action taken:", result?.result?.action_taken);
        
        if (result?.result?.page_classification === 'bad') {
          var actionTaken = result.result.action_taken || 'page_blocked';
          
          // Block this URL for future visits (in-memory)
          blockBadUrl(tab.url, result.result.page_reasons);
          
          if (actionTaken === 'page_blocked') {
            // High risk domain - block entire page
            console.log("HIGH RISK DOMAIN - Blocking entire page");
            chrome.tabs.update(tabId, { url: 'blocked.html?url=' + encodeURIComponent(tab.url) });
          } else {
            // High reputation domain - remove offending content only
            console.log("HIGH REPUTATION DOMAIN - Removing bad content only");
            // Send message to remove bad elements
            chrome.scripting.executeScript({
              target: { tabId },
              files: ['content.js']
            }).then(() => {
              return chrome.tabs.sendMessage(tabId, {
                type: "remove_all_bad",
                badImages: result.result.image_analysis?.bad_image_elements || [],
                badText: result.result.text_analysis?.bad_text_elements || []
              });
            }).catch(err => {
              console.log("Content removal error:", err.message);
            });
          }
        }
      }).catch(err => {
        console.log("Auto-analysis error:", err.message);
      });
    }, 500);
  }
});
