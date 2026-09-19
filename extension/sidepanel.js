/**
 * StudyTube Side Panel Controller
 * Handles tab monitoring, server health checks, iframe embedding, and seek message forwarding.
 */

const SERVER_BASE = "http://localhost:8501";
const HEALTH_URL = `${SERVER_BASE}/_stcore/health`;

// DOM Elements
const appFrame = document.getElementById("app-frame");
const viewLoading = document.getElementById("view-loading");
const viewNotYouTube = document.getElementById("view-not-youtube");
const viewOffline = document.getElementById("view-offline");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const btnRetry = document.getElementById("btn-retry");
const btnOpenYouTube = document.getElementById("btn-open-youtube");

let currentLoadedVideoId = null;

// Update status badge UI
function setStatus(state, label) {
  statusDot.className = `dot ${state}`;
  statusText.textContent = label;
}

// Switch active view
function showView(viewName) {
  viewLoading.classList.remove("active");
  viewNotYouTube.classList.remove("active");
  viewOffline.classList.remove("active");
  appFrame.classList.remove("visible");

  if (viewName === "iframe") {
    appFrame.classList.add("visible");
  } else if (viewName === "not_youtube") {
    viewNotYouTube.classList.add("active");
  } else if (viewName === "offline") {
    viewOffline.classList.add("active");
  } else {
    viewLoading.classList.add("active");
  }
}

// Extract video ID from URL
function extractVideoId(url) {
  if (!url) return null;
  try {
    const parsed = new URL(url);
    if (parsed.hostname.includes("youtube.com") && parsed.pathname === "/watch") {
      return parsed.searchParams.get("v");
    }
    if (parsed.hostname === "youtu.be") {
      return parsed.pathname.slice(1);
    }
  } catch (e) {
    return null;
  }
  return null;
}

// Ping the Streamlit server health endpoint
async function checkServerHealth() {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 2000);

  try {
    const res = await fetch(HEALTH_URL, {
      method: "GET",
      signal: controller.signal,
      cache: "no-store",
    });
    clearTimeout(timeoutId);
    return res.ok || res.status === 200;
  } catch (err) {
    clearTimeout(timeoutId);
    return false;
  }
}

// Sync side panel with currently active browser tab
async function syncWithActiveTab() {
  setStatus("checking", "Checking...");

  const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!activeTab || !activeTab.url) {
    setStatus("error", "No Tab");
    showView("not_youtube");
    return;
  }

  const videoId = extractVideoId(activeTab.url);
  if (!videoId) {
    setStatus("error", "Not YouTube");
    showView("not_youtube");
    currentLoadedVideoId = null;
    return;
  }

  // Check if Streamlit backend is reachable
  const isServerUp = await checkServerHealth();
  if (!isServerUp) {
    setStatus("error", "Server Offline");
    showView("offline");
    return;
  }

  setStatus("connected", "Ready");

  // Load or update iframe only if video changed
  if (currentLoadedVideoId !== videoId) {
    currentLoadedVideoId = videoId;
    const targetUrl = `${SERVER_BASE}/?embed=true&video_url=${encodeURIComponent(activeTab.url)}`;
    console.log("[StudyTube SidePanel] Embedding Streamlit for video:", videoId);
    appFrame.src = targetUrl;
  }

  showView("iframe");
}

// Listen for seek messages forwarded from the Streamlit iframe
window.addEventListener("message", async (event) => {
  if (event.data?.type === "STUDYTUBE_SEEK") {
    const seconds = event.data.seconds;
    console.log("[StudyTube SidePanel] Received seek command for seconds:", seconds);

    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.id) {
      try {
        await chrome.tabs.sendMessage(tab.id, {
          type: "SEEK_TO",
          seconds: seconds,
        });
      } catch (err) {
        console.warn("[StudyTube SidePanel] Failed to message YouTube tab:", err);
      }
    }
  }
});

// Listen for tab updates broadcast by background.js
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === "STUDYTUBE_TAB_UPDATED") {
    syncWithActiveTab();
  }
});

// Button events
btnRetry.addEventListener("click", () => {
  syncWithActiveTab();
});

btnOpenYouTube.addEventListener("click", () => {
  chrome.tabs.create({ url: "https://www.youtube.com" });
});

// Initial boot
document.addEventListener("DOMContentLoaded", () => {
  syncWithActiveTab();
});
