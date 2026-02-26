/**
 * Content Script for Child Safety Monitor Extension
 * Runs on monitored websites to analyze content
 */

console.log("SafeNest: Content script loaded!");

let chatObserver = null;
let inputMonitorIntervalId = null;
let reinitTimeoutId = null;

// Pattern detection (client-side quick check before sending to OS)
const RISK_PATTERNS = {
  personal_info: {
    patterns: [
      /\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/g, // Phone numbers
      /\b\d+\s+\w+\s+(street|road|ave|avenue|blvd|boulevard)\b/gi, // Addresses
      /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi, // Emails
    ],
    level: "high",
    reason: "sharing_personal_info",
  },
  unsafe_requests: {
    patterns: [
      /send\s+(me\s+)?(your\s+)?(pic|picture|photo)/gi,
      /(what'?s|where'?s)\s+your\s+(address|location|school)/gi,
      /meet\s+me\s+(alone|secretly)/gi,
    ],
    level: "high",
    reason: "unsafe_request",
  },
  bullying: {
    patterns: [
      /(kill|hurt)\s+yourself/gi,
      /you'?re\s+(worthless|stupid|ugly|fat)/gi,
      /nobody\s+likes\s+you/gi,
    ],
    level: "medium",
    reason: "bullying",
  },
};

// Site-specific selectors for chat/message areas
const SITE_SELECTORS = {
  "youtube.com": {
    chat: "#message, #content-text, .yt-live-chat-text-message-renderer, #comment-content",
    input:
      "#contenteditable-root, #simplebox-placeholder, ytd-commentbox #contenteditable-root, [contenteditable='true'][aria-label*='comment' i]",
  },
  "instagram.com": {
    chat: '[role="textbox"], .x1lliihq, ._a9zs',
    input: '[contenteditable="true"]',
  },
  "discord.com": {
    chat: '.messageContent-2qWWxC, [class*="messageContent"]',
    input: '[class*="textArea"], [role="textbox"]',
  },
  "roblox.com": {
    chat: '.text-chat-message, [class*="Message"]',
    input: '.text-chat-input, [class*="InputBox"]',
  },
  "tiktok.com": {
    chat: '[data-e2e="comment-level-1"], .comment-text',
    input: '[data-e2e="comment-input"]',
  },
};

// Get current site configuration
const hostname = window.location.hostname;
const siteConfig = Object.entries(SITE_SELECTORS).find(([site]) =>
  hostname.includes(site),
)?.[1];

console.log("SafeNest: Script running on:", hostname);
console.log("SafeNest: Site config found:", siteConfig);

/**
 * Analyze text for risk patterns
 */
function analyzeText(text) {
  const risks = [];

  for (const [category, config] of Object.entries(RISK_PATTERNS)) {
    for (const pattern of config.patterns) {
      pattern.lastIndex = 0;
      if (pattern.test(text)) {
        risks.push({
          category,
          level: config.level,
          reason: config.reason,
        });
        break; // One match per category is enough
      }
    }
  }

  return risks;
}

/**
 * Send analysis to background script
 */
function sendToMonitor(text, risks) {
  if (risks.length === 0) return;

  const highestRisk = risks.reduce((max, risk) => {
    const levels = { low: 1, medium: 2, high: 3 };
    return levels[risk.level] > levels[max.level] ? risk : max;
  }, risks[0]);

  chrome.runtime.sendMessage(
    {
      type: "analyze_content",
      riskLevel: highestRisk.level,
      reason: highestRisk.reason,
      categories: risks.map((r) => r.category),
      site: hostname,
      timestamp: Date.now(),
    },
    (response) => {
      if (chrome.runtime.lastError) {
        console.error("Failed to send to monitor:", chrome.runtime.lastError);
      }
    },
  );
}

/**
 * Monitor chat messages
 */
function monitorMessages() {
  if (!siteConfig) return;

  if (chatObserver) {
    chatObserver.disconnect();
  }

  const chatSelector = siteConfig.chat;

  // Observe new messages
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeType === Node.ELEMENT_NODE) {
          const messages = node.querySelectorAll
            ? node.querySelectorAll(chatSelector)
            : node.matches?.(chatSelector)
              ? [node]
              : [];

          messages.forEach((msgElement) => {
            const text = msgElement.textContent || msgElement.innerText;
            if (text && text.trim()) {
              const risks = analyzeText(text);
              if (risks.length > 0) {
                sendToMonitor(text, risks);
              }
            }
          });
        }
      });
    });
  });

  // Start observing
  observer.observe(document.body, {
    childList: true,
    subtree: true,
  });

  chatObserver = observer;

  // Also check existing messages
  document.querySelectorAll(chatSelector).forEach((msgElement) => {
    const text = msgElement.textContent || msgElement.innerText;
    if (text && text.trim()) {
      const risks = analyzeText(text);
      if (risks.length > 0) {
        sendToMonitor(text, risks);
      }
    }
  });
}

/**
 * Monitor input fields to check outgoing messages
 */
function monitorInput() {
  if (!siteConfig) {
    console.log("SafeNest: No site config, skipping input monitoring");
    return;
  }

  const inputSelector = siteConfig.input;
  console.log("SafeNest: Looking for inputs with selector:", inputSelector);

  // Find input fields
  const checkInputs = () => {
    const inputs = document.querySelectorAll(inputSelector);
    console.log("SafeNest: Found inputs:", inputs.length, inputs);

    inputs.forEach((input, index) => {
      console.log("SafeNest attached:", input.dataset.monitorAttached);

      if (input.dataset.monitorAttached) return;
      input.dataset.monitorAttached = "true";

      console.log("SafeNest: Attaching listeners to input", index);

      // Monitor input/paste events
      ["input", "paste", "keyup", "textInput"].forEach((eventType) => {
        input.addEventListener(
          eventType,
          (e) => {
            console.log("SafeNest: Event detected:", eventType);
            setTimeout(() => {
              // For contenteditable, use textContent or innerText
              const text =
                input.textContent || input.innerText || input.value || "";
              console.log("SafeNest: Current text:", text);

              if (text && text.trim()) {
                const risks = analyzeText(text);
                console.log("SafeNest: Risks found:", risks);

                if (risks.length > 0 && risks.some((r) => r.level === "high")) {
                  // Warn user before sending
                  showWarning(input, risks);
                  sendToMonitor(text, risks);
                }
              }
            }, 100);
          },
          true,
        ); // Add capture phase
      });
      // Monitor submit
      const form = input.closest("form");
      if (form && !form.dataset.monitorAttached) {
        form.dataset.monitorAttached = "true";
        console.log("SafeNest: Attaching form listener");
        form.addEventListener(
          "submit",
          (e) => {
            const text = input.textContent || input.value;
            console.log("SafeNest: Form submit, text:", text);
            if (text && text.trim()) {
              const risks = analyzeText(text);

              if (risks.length > 0 && risks.some((r) => r.level === "high")) {
                e.preventDefault();
                e.stopPropagation();
                showBlockedMessage(input);
                sendToMonitor(text, risks);
                return false;
              }
            }
          },
          true,
        );
      }
    });
  };

  // Check initially and on mutations
  checkInputs();

  if (!inputMonitorIntervalId) {
    inputMonitorIntervalId = setInterval(checkInputs, 2000);
  }
}

/**
 * Show warning to user
 */
function showWarning(inputElement, risks) {
  // Remove existing warning
  const existing = document.getElementById("safety-monitor-warning");
  if (existing) existing.remove();

  const warning = document.createElement("div");
  warning.id = "safety-monitor-warning";
  warning.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: #ff9800;
    color: white;
    padding: 15px 20px;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    z-index: 999999;
    max-width: 300px;
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 14px;
  `;

  const categories = [...new Set(risks.map((r) => r.category))].join(", ");
  warning.innerHTML = `
    <strong>⚠️ Safety Warning</strong><br>
    This message may contain: ${categories}<br>
    <small>Your parent will be notified.</small>
  `;

  document.body.appendChild(warning);

  setTimeout(() => warning.remove(), 5000);
}

/**
 * Show blocked message
 */
function showBlockedMessage(inputElement) {
  const blocked = document.createElement("div");
  blocked.style.cssText = `
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: #f44336;
    color: white;
    padding: 30px;
    border-radius: 12px;
    box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    z-index: 999999;
    text-align: center;
    font-family: system-ui, -apple-system, sans-serif;
  `;

  blocked.innerHTML = `
    <h2 style="margin: 0 0 15px 0;">🛡️ Message Blocked</h2>
    <p style="margin: 0 0 15px 0;">
      This message was blocked for your safety.<br>
      Your parent has been notified.
    </p>
    <button id="safety-monitor-ok" style="
      background: white;
      color: #f44336;
      border: none;
      padding: 10px 30px;
      border-radius: 6px;
      font-size: 16px;
      font-weight: bold;
      cursor: pointer;
    ">OK</button>
  `;

  document.body.appendChild(blocked);

  document.getElementById("safety-monitor-ok").addEventListener("click", () => {
    blocked.remove();
    inputElement.textContent = "";
    inputElement.value = "";
  });
}

/**
 * Handle messages from background script
 */
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log("Content script received message:", request.type, request);

  if (request.type === "block_content") {
    console.log("Blocking content:", request.reason);
    
    if (request.elements && request.elements.length > 0) {
      request.elements.forEach(element => {
        removeElementByXPath(element.xpath);
      });
    }
    
    sendResponse({ received: true, blocked: true });
    return true;
  }
  
  if (request.type === "remove_bad_images") {
    console.log("Removing bad images:", request.elements?.length || 0);
    
    if (request.elements && request.elements.length > 0) {
      request.elements.forEach(element => {
        if (element.xpath) {
          removeElementByXPath(element.xpath);
        }
      });
    }
    
    sendResponse({ received: true, blocked: true });
    return true;
  }
  
  if (request.type === "remove_bad_text") {
    console.log("Removing bad text elements:", request.elements?.length || 0);
    
    if (request.elements && request.elements.length > 0) {
      request.elements.forEach(element => {
        if (element.xpath) {
          removeElementByXPath(element.xpath);
        }
      });
    }
    
    sendResponse({ received: true, blocked: true });
    return true;
  }
  
  if (request.type === "analyze_page_dom") {
    console.log("Analyzing page DOM...");

    const parseSrcsetUrls = (srcset) => {
      if (!srcset || typeof srcset !== 'string') return [];
      return srcset
        .split(',')
        .map((part) => part.trim().split(' ')[0])
        .filter(Boolean);
    };
    
    // Get all images with their current URLs and positions
    const images = [];
    const allImgElements = document.querySelectorAll('img, video[poster], [data-src], [data-lazy-src], source[srcset], picture source[srcset], img[data-srcset]');
    allImgElements.forEach((img, idx) => {
      const candidateSrcs = [
        img.src,
        img.currentSrc,
        img.getAttribute('data-src'),
        img.getAttribute('data-lazy-src'),
        img.poster,
        ...parseSrcsetUrls(img.getAttribute('srcset')),
        ...parseSrcsetUrls(img.getAttribute('data-srcset')),
      ].filter(Boolean);

      const width = img.naturalWidth || img.width || 0;
      const height = img.naturalHeight || img.height || 0;

      candidateSrcs.forEach((src) => {
        if (!src || src.startsWith('data:') || src.includes('chrome-extension')) return;

        // Skip small images/icons - check naturalWidth and naturalHeight if available
        // Skip icons and small images (less than 100x100)
        if (width > 100 || height > 100) {
          images.push({
            idx: idx,
            tag: img.tagName.toLowerCase(),
            src: src,
            id: img.id || '',
            class: img.className || '',
            alt: img.alt || '',
            width: width,
            height: height,
          });
        }
      });
    });
    
    const prioritizedImages = images
      .sort((a, b) => ((b.width || 0) * (b.height || 0)) - ((a.width || 0) * (a.height || 0)))
      .slice(0, 24);

    // Get text content from main elements
    const textSelectors = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'li', 'td', 'th'];
    const textElements = [];
    textSelectors.forEach(selector => {
      document.querySelectorAll(selector).forEach((el, idx) => {
        const text = el.textContent?.trim() || '';
        if (text.length > 20 && text.length < 1000) {
          textElements.push({
            tag: selector,
            text: text,
            id: el.id || '',
            class: el.className || '',
          });
        }
      });
    });
    
    const cappedTextElements = textElements.slice(0, 120);

    const domData = {
      url: window.location.href,
      html: "",
      images: prioritizedImages,
      textElements: cappedTextElements,
    };
    
    console.log("Sending DOM data:", { images: prioritizedImages.length, textElements: cappedTextElements.length });
    sendResponse({ success: true, domData: domData });
    return true;
  }
  
  if (request.type === "remove_all_bad") {
    console.log("Received remove_all_bad:", request);
    console.log("Bad images count:", request.badImages?.length);
    console.log("Bad text count:", request.badText?.length);
    
    let removedCount = 0;
    const imageAction = request.imageAction || "remove";
    const replacementImage = request.replacementImage || chrome.runtime.getURL("assets/images/stop.png");
    
    // For images, try multiple methods to find and remove them
    if (request.badImages && request.badImages.length > 0) {
      // Normalize URL for comparison
      const normalizeUrl = (url) => {
        try {
          const urlObj = new URL(url);
          return urlObj.pathname + urlObj.search;
        } catch {
          return url;
        }
      };

      // Build one-time media index to avoid repeated full-page scans for each bad image
      const allImages = Array.from(document.querySelectorAll('img, video, [data-src], [data-lazy-src], source[srcset]'));
      const exactUrlMap = new Map();
      const pathMap = new Map();
      const filenameMap = new Map();

      const addToMap = (map, key, element) => {
        if (!key) return;
        const normalizedKey = key.toString().trim();
        if (!normalizedKey) return;
        const existing = map.get(normalizedKey);
        if (existing) {
          existing.push(element);
        } else {
          map.set(normalizedKey, [element]);
        }
      };

      allImages.forEach((img) => {
        const srcs = [
          img.src,
          img.currentSrc,
          img.getAttribute('data-src'),
          img.getAttribute('data-lazy-src'),
          img.getAttribute('data-srcset')?.split(' ')[0],
          img.poster,
          img.querySelector?.('source[src]')?.src,
        ].filter(Boolean);

        srcs.forEach((src) => {
          addToMap(exactUrlMap, src, img);

          try {
            const srcPath = normalizeUrl(src);
            addToMap(pathMap, srcPath, img);
          } catch {}

          const filename = src.split('/').pop()?.split('?')[0];
          if (filename && filename.length > 5) {
            addToMap(filenameMap, filename, img);
          }
        });
      });

      request.badImages.forEach((element, idx) => {
        const imageUrl = element.url;
        const imageIdx = typeof element.idx === 'number' ? element.idx : -1;
        const elementInfo = element.element_info || {};
        const cssSelector = elementInfo.css_selector;
        const tag = elementInfo.tag || 'img';
        
        console.log(`Processing bad image ${idx}: ${imageUrl}`);
                
        const serverPath = normalizeUrl(imageUrl);
        
        let found = false;

        // Fast exact element index match from dom_analysis capture
        if (!found && imageIdx >= 0 && imageIdx < allImages.length) {
          const indexedEl = allImages[imageIdx];
          if (indexedEl) {
            console.log(`Found image by index: ${imageIdx}`);
            if (imageAction === "replace_with_warning") {
              replaceWithWarningImage(indexedEl, replacementImage);
            } else {
              removeUnsafeMediaElement(indexedEl);
            }
            removedCount++;
            found = true;
          }
        }

        const hideCandidates = (candidates, reason) => {
          if (!candidates || candidates.length === 0) return false;
          candidates.forEach((candidate) => {
            console.log(`Found image by ${reason}`);
            if (imageAction === "replace_with_warning") {
              replaceWithWarningImage(candidate, replacementImage);
            } else {
              removeUnsafeMediaElement(candidate);
            }
            removedCount++;
          });
          return true;
        };

        // Fast exact URL match
        if (!found) {
          found = hideCandidates(exactUrlMap.get(imageUrl), `exact URL: ${imageUrl}`);
        }

        // Fast path match
        if (!found) {
          found = hideCandidates(pathMap.get(serverPath), `path: ${serverPath}`);
        }

        // Fast filename match
        if (!found) {
          const serverFilename = imageUrl.split('/').pop()?.split('?')[0];
          if (serverFilename && serverFilename.length > 5) {
            found = hideCandidates(filenameMap.get(serverFilename), `filename: ${serverFilename}`);
          }
        }
        
        // Try CSS selector if provided and not just a generic tag
        if (!found && cssSelector && cssSelector.length > 3 && !cssSelector.match(/^img$/)) {
          try {
            const els = document.querySelectorAll(cssSelector);
            for (const el of els) {
              console.log(`Found image by CSS: ${cssSelector}`);
              if (imageAction === "replace_with_warning") {
                replaceWithWarningImage(el, replacementImage);
              } else {
                removeUnsafeMediaElement(el);
              }
              removedCount++;
              found = true;
            }
          } catch (e) {}
        }
        
        // Try matching by element attributes (id, class, data attributes)
        if (!found) {
          const elId = elementInfo.id;
          const elClass = elementInfo.class;
          const dataAttrs = elementInfo.data_attributes || {};
          
          if (elId) {
            const el = document.getElementById(elId);
            if (el && (el.tagName === 'IMG' || el.tagName === 'VIDEO')) {
              console.log(`Found image by ID: ${elId}`);
              if (imageAction === "replace_with_warning") {
                replaceWithWarningImage(el, replacementImage);
              } else {
                removeUnsafeMediaElement(el);
              }
              removedCount++;
              found = true;
            }
          }
          
          // Try class matching
          if (!found && elClass) {
            const classes = elClass.split(' ').filter(c => c.length > 2);
            for (const cls of classes) {
              const els = document.querySelectorAll(`${tag}.${cls}`);
              for (const el of els) {
                console.log(`Found image by class: ${cls}`);
                if (imageAction === "replace_with_warning") {
                  replaceWithWarningImage(el, replacementImage);
                } else {
                  removeUnsafeMediaElement(el);
                }
                removedCount++;
                found = true;
              }
              if (found) break;
            }
          }
        }
        
        if (!found) {
          console.log(`Image not found: ${imageUrl}`);
        }
      });
    }
    
    // For text elements, try XPath, CSS selector, and text content matching
    if (request.badText && request.badText.length > 0) {
      request.badText.forEach((element, idx) => {
        const xpath = element.element_info?.xpath;
        const cssSelector = element.element_info?.css_selector;
        const tag = element.element_info?.tag || 'div';
        const elId = element.element_info?.id;
        const elClass = element.element_info?.class;
        const text = element.text;
        
        let removed = false;
        
        // Try XPath first
        if (xpath) {
          removed = removeElementByXPath(xpath);
        }
        
        // Try CSS selector
        if (!removed && cssSelector && cssSelector.length > 3) {
          try {
            const els = document.querySelectorAll(cssSelector);
            for (const el of els) {
              // If we have text content, check if it matches
              if (text && el.textContent && el.textContent.includes(text.substring(0, 50))) {
                console.log(`Found text element by CSS with text match: ${cssSelector}`);
                hideElement(el);
                removedCount++;
                removed = true;
                break;
              } else if (!text) {
                console.log(`Found text element by CSS: ${cssSelector}`);
                hideElement(el);
                removedCount++;
                removed = true;
                break;
              }
            }
          } catch (e) {}
        }
        
        // Try matching by ID
        if (!removed && elId) {
          const el = document.getElementById(elId);
          if (el) {
            console.log(`Found text element by ID: ${elId}`);
            hideElement(el);
            removedCount++;
            removed = true;
          }
        }
        
        // Try matching by class
        if (!removed && elClass) {
          const classes = elClass.split(' ').filter(c => c.length > 2);
          for (const cls of classes) {
            const els = document.querySelectorAll(`${tag}.${cls}`);
            for (const el of els) {
              // Check text content match
              if (text && el.textContent && el.textContent.toLowerCase().includes(text.toLowerCase().substring(0, 30))) {
                console.log(`Found text element by class with text match: ${cls}`);
                hideElement(el);
                removedCount++;
                removed = true;
                break;
              }
            }
            if (removed) break;
          }
        }
        
        // Try text content matching (last resort)
        if (!removed && text && text.length > 10) {
          const searchText = text.substring(0, 50).toLowerCase();
          const allElements = document.querySelectorAll('p, span, div, a, li, td, th, h1, h2, h3, h4, h5, h6');
          for (const el of allElements) {
            if (el.textContent && el.textContent.toLowerCase().includes(searchText)) {
              console.log(`Found text element by content: ${searchText}`);
              hideElement(el);
              removedCount++;
              removed = true;
              break;
            }
          }
        }
        
        console.log(`Text element ${idx}: ${removed ? 'removed' : 'not found'}`);
      });
    }
    
    console.log(`Total removed: ${removedCount}`);
    sendResponse({ received: true, blocked: true, removed: removedCount });
    return true;
  }

  sendResponse({ received: true });
  return true;
});

/**
 * Hide an element safely
 */
function hideElement(element) {
  if (!element) return;
  element.style.display = 'none';
  element.style.visibility = 'hidden';
  element.style.opacity = '0';
  element.setAttribute('data-safety-blocked', 'true');
}

function removeUnsafeMediaElement(element) {
  if (!element) return;

  hideElement(element);

  const wrapperLink = element.closest && element.closest('a');
  if (!wrapperLink) return;

  const wrapperText = (wrapperLink.textContent || '').trim();
  const mediaChildren = wrapperLink.querySelectorAll('img, video, picture, source').length;

  if (!wrapperText || mediaChildren > 0) {
    wrapperLink.style.pointerEvents = 'none';
    wrapperLink.style.cursor = 'default';
    wrapperLink.removeAttribute('href');
    wrapperLink.setAttribute('data-safety-blocked', 'true');
  }
}

function replaceWithWarningImage(element, warningImageUrl) {
  if (!element) return;

  const tag = (element.tagName || '').toLowerCase();
  element.setAttribute('data-safety-blocked', 'true');

  if (tag === 'img') {
    element.dataset.safetyOriginalSrc = element.src || '';
    element.src = warningImageUrl;
    element.srcset = '';
    element.style.filter = 'blur(4px)';
    element.style.objectFit = 'cover';
    element.alt = 'Blocked unsafe image';
    return;
  }

  if (tag === 'video') {
    if (typeof element.pause === 'function') {
      element.pause();
    }
    element.poster = warningImageUrl;
    element.style.filter = 'blur(4px)';
    return;
  }

  element.style.backgroundImage = `url('${warningImageUrl}')`;
  element.style.backgroundSize = 'cover';
  element.style.backgroundPosition = 'center';
  element.style.filter = 'blur(4px)';
  if (!element.style.minHeight) {
    element.style.minHeight = '120px';
  }
}

/**
 * Remove an element by its XPath
 */
function removeElementByXPath(xpath) {
  try {
    const result = document.evaluate(
      xpath,
      document,
      null,
      XPathResult.FIRST_ORDERED_NODE_TYPE,
      null
    );
    
    if (result.singleNodeValue) {
      const element = result.singleNodeValue;
      element.style.display = 'none';
      element.style.visibility = 'hidden';
      element.style.opacity = '0';
      element.setAttribute('data-safety-blocked', 'true');
      console.log("Blocked element:", xpath);
      return true;
    } else {
      console.log("Element not found for xpath:", xpath);
      return false;
    }
  } catch (error) {
    console.error("Error removing element:", error);
    return false;
  }
}

/**
 * Initialize monitoring
 */
console.log("Child Safety Monitor: Content script loaded");

// Wait for page to be ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    monitorMessages();
    monitorInput();
  });
} else {
  monitorMessages();
  monitorInput();
}

// Re-initialize on navigation (for SPAs)
let lastUrl = location.href;
new MutationObserver(() => {
  const url = location.href;
  if (url !== lastUrl) {
    lastUrl = url;
    console.log("SafeNest: URL changed, reinitializing...");

    if (reinitTimeoutId) {
      clearTimeout(reinitTimeoutId);
    }

    reinitTimeoutId = setTimeout(() => {
      monitorMessages();
      monitorInput();
    }, 1000);
  }
}).observe(document, { subtree: true, childList: true });
