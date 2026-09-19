/**
 * StudyTube Content Script
 * Injected on YouTube pages to control video playback (seek, play, and status).
 */

console.log("[StudyTube Content Script] Active on YouTube.");

// Flash a mini HUD toast on the YouTube player
function showSeekToast(seconds) {
  const existing = document.getElementById("studytube-seek-toast");
  if (existing) existing.remove();

  const totalSecs = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(totalSecs / 60);
  const secs = totalSecs % 60;
  const timeFormatted = `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;

  const toast = document.createElement("div");
  toast.id = "studytube-seek-toast";
  toast.textContent = `⏱️ StudyTube: Jumped to ${timeFormatted}`;
  Object.assign(toast.style, {
    position: "fixed",
    top: "60px",
    right: "24px",
    backgroundColor: "rgba(15, 23, 42, 0.92)",
    color: "#FFFFFF",
    padding: "10px 16px",
    borderRadius: "8px",
    fontSize: "14px",
    fontWeight: "600",
    zIndex: "9999999",
    boxShadow: "0 4px 12px rgba(0, 0, 0, 0.3)",
    transition: "opacity 0.25s ease-in-out",
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    pointerEvents: "none",
  });

  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 250);
  }, 1600);
}

// Listen for commands from the side panel
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "SEEK_TO") {
    const seconds = Number(message.seconds);
    const video = document.querySelector("video");

    if (video && !isNaN(seconds)) {
      video.currentTime = seconds;
      video.play().catch(() => {
        // Autoplay may be restricted in some cases; currentTime is still updated
      });
      showSeekToast(seconds);
      sendResponse({ success: true, currentTime: video.currentTime });
    } else {
      console.warn("[StudyTube Content] Could not find video player element.");
      sendResponse({ success: false, error: "HTML5 video element not found." });
    }
  } else if (message.type === "GET_VIDEO_STATE") {
    const video = document.querySelector("video");
    sendResponse({
      available: !!video,
      currentTime: video ? video.currentTime : 0,
      paused: video ? video.paused : true,
    });
  }

  return true; // Keep message channel open
});
