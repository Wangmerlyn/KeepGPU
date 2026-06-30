export const AUTO_REFRESH_INTERVAL_MS = 10000

export function canRunAutoRefresh(autoRefresh, visibilityState = "visible") {
  return autoRefresh && visibilityState !== "hidden"
}

export function canReuseInFlightRefresh(inFlightRefresh, afterMutation = false) {
  return Boolean(inFlightRefresh) && !afterMutation
}

export function formatRefreshMode(autoRefresh, visibilityState = "visible") {
  if (!autoRefresh) {
    return "manual refresh"
  }
  if (visibilityState === "hidden") {
    return "auto paused"
  }
  return `auto ${AUTO_REFRESH_INTERVAL_MS / 1000}s`
}
