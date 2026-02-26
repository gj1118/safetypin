const params = new URLSearchParams(window.location.search);
const view = params.get("view");

function closeTab() {
  chrome.tabs.close();
}

function renderBlockedPage() {
  const app = document.getElementById("app");
  app.innerHTML = `
    <div class="container">
      <img class="hero-art" src="./assets/images/stop.png" alt="Blocked by SafetyPin" />
      <div class="pill"><span class="dot"></span>Safety Alert</div>
      <h1>Page blocked for safety</h1>
      <p>
        SafetyPin blocked this page because it appears to contain unsafe or
        inappropriate content.
      </p>
      <div class="url" id="blockedUrl"></div>
      <div class="actions">
        <button id="closeBtn">Close this page</button>
        <p class="subtext">You can return to a safer page anytime.</p>
      </div>
    </div>
  `;

  document.getElementById("blockedUrl").textContent = params.get("url") || "Unknown";
  document.getElementById("closeBtn").addEventListener("click", closeTab);
}

function createDetailsItem(section, item, index) {
  const wrapper = document.createElement("div");
  wrapper.className = "details-item";

  const heading = document.createElement("h3");
  heading.textContent = `${section === "images" ? "Image" : "Text"} #${index + 1}`;
  wrapper.appendChild(heading);

  const confidence = document.createElement("p");
  confidence.innerHTML = `<strong>Confidence:</strong> ${(Number(item.confidence || 0) * 100).toFixed(1)}%`;
  wrapper.appendChild(confidence);

  if (section === "images") {
    const url = document.createElement("p");
    url.innerHTML = `<strong>URL:</strong> <span class="mono"></span>`;
    url.querySelector("span").textContent = item.url || "unknown";
    wrapper.appendChild(url);
  } else {
    const text = document.createElement("p");
    text.innerHTML = `<strong>Snippet:</strong> <span class="mono"></span>`;
    text.querySelector("span").textContent = item.text || "(empty)";
    wrapper.appendChild(text);
  }

  if (item.reason) {
    const reason = document.createElement("p");
    reason.innerHTML = `<strong>Reason:</strong> ${item.reason}`;
    wrapper.appendChild(reason);
  }

  const selector = item.element_info?.css_selector;
  if (selector) {
    const selectorP = document.createElement("p");
    selectorP.innerHTML = `<strong>Element:</strong> <span class="mono"></span>`;
    selectorP.querySelector("span").textContent = selector;
    wrapper.appendChild(selectorP);
  }

  return wrapper;
}

function renderDetailsPage(report) {
  const app = document.getElementById("app");
  const section = report.section === "text" ? "text" : "images";
  const title = section === "images" ? "Blocked images details" : "Blocked text details";

  app.innerHTML = `
    <div class="details-shell">
      <div class="details-head">
        <div>
          <h1 class="details-title">${title}</h1>
          <p class="details-meta" id="detailsMeta"></p>
        </div>
        <button id="closeBtn">Close this page</button>
      </div>
      <div class="details-list" id="detailsList"></div>
    </div>
  `;

  const detailsMeta = document.getElementById("detailsMeta");
  const itemCount = report.items?.length || 0;
  detailsMeta.textContent = `${itemCount} item${itemCount === 1 ? "" : "s"} • ${report.domain || "unknown domain"}`;

  const detailsList = document.getElementById("detailsList");
  if (!itemCount) {
    const empty = document.createElement("p");
    empty.className = "subtext";
    empty.textContent = "No blocked items found for this section.";
    detailsList.appendChild(empty);
  } else {
    report.items.forEach((item, index) => {
      detailsList.appendChild(createDetailsItem(section, item, index));
    });
  }

  document.getElementById("closeBtn").addEventListener("click", closeTab);
}

function renderMissingDetails() {
  const app = document.getElementById("app");
  app.innerHTML = `
    <div class="container">
      <div class="pill"><span class="dot"></span>SafetyPin</div>
      <h1>Details are unavailable</h1>
      <p>The blocked-item details could not be loaded. Please run a new scan and try again.</p>
      <div class="actions">
        <button id="closeBtn">Close this page</button>
      </div>
    </div>
  `;
  document.getElementById("closeBtn").addEventListener("click", closeTab);
}

if (view === "details") {
  const reportId = params.get("id");
  if (!reportId) {
    renderMissingDetails();
  } else {
    chrome.storage.local.get([reportId], (result) => {
      const report = result[reportId];
      if (!report) {
        renderMissingDetails();
        return;
      }

      renderDetailsPage(report);
      chrome.storage.local.remove(reportId);
    });
  }
} else {
  renderBlockedPage();
}
