/**
 * StudyTube Background Service Worker (Manifest V3)
 * Manages side panel behavior, tab URL changes, and video sync.
 */

// Enable side panel to open when clicking the action icon
chrome.runtime.onInstalled.addListener(() => {
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch((err) => {
    console.warn("[StudyTube SW] setPanelBehavior error:", err);
  });
  console.log("[StudyTube SW] Extension installed successfully.");
});

// Fallback action click listener
chrome.action.onClicked.addListener(async (tab) => {
  if (tab?.windowId) {
    try {
      await chrome.sidePanel.open({ windowId: tab.windowId });
    } catch (err) {
      console.warn("[StudyTube SW] Could not open side panel:", err);
    }
  }
});

// Helper to check if URL is a YouTube watch page
function isYouTubeWatch(url) {
  if (!url) return false;
  return url.includes("youtube.com/watch") || url.includes("youtu.be/");
}

// Track tab updates (handles YouTube SPA navigation like history.pushState)
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  const currentUrl = changeInfo.url || tab.url;
  if (!currentUrl) return;

  // Only broadcast if the active tab in current window updated
  const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (activeTab && activeTab.id === tabId) {
    await chrome.storage.session.set({ activeUrl: currentUrl });
    // Notify side panel about potential URL change
    chrome.runtime.sendMessage({
      type: "STUDYTUBE_TAB_UPDATED",
      url: currentUrl,
      isYouTube: isYouTubeWatch(currentUrl),
    }).catch(() => {
      // Side panel might not be open; ignore error
    });
  }
});

// Track tab switches
chrome.tabs.onActivated.addListener(async (activeInfo) => {
  try {
    const tab = await chrome.tabs.get(activeInfo.tabId);
    if (tab?.url) {
      await chrome.storage.session.set({ activeUrl: tab.url });
      chrome.runtime.sendMessage({
        type: "STUDYTUBE_TAB_UPDATED",
        url: tab.url,
        isYouTube: isYouTubeWatch(tab.url),
      }).catch(() => {});
    }
  } catch (err) {
    console.debug("[StudyTube SW] onActivated error:", err);
  }
});
