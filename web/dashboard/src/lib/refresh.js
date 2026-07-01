export const AUTO_REFRESH_INTERVAL_MS = 10000

export function canRunAutoRefresh(autoRefresh, visibilityState = "visible") {
  return autoRefresh && visibilityState !== "hidden"
}

export function canReuseInFlightRefresh(inFlightRefresh, afterMutation = false) {
  return Boolean(inFlightRefresh) && !afterMutation
}

export function formatRefreshWarningMessage(error) {
  const message =
    error instanceof Error ? error.message : String(error ?? "unknown error")
  return `Refresh warning: ${message || "unknown error"}`
}

export async function fetchDashboardPayloads(requestJson) {
  const [gpuResult, sessionResult] = await Promise.allSettled([
    requestJson("GET", "/api/gpus"),
    requestJson("GET", "/api/sessions")
  ])

  return {
    gpus: gpuResult.status === "fulfilled" ? gpuResult.value?.gpus ?? [] : null,
    sessions:
      sessionResult.status === "fulfilled"
        ? sessionResult.value?.active_jobs ?? []
        : null,
    warning:
      gpuResult.status === "rejected"
        ? formatRefreshWarningMessage(gpuResult.reason)
        : sessionResult.status === "rejected"
          ? formatRefreshWarningMessage(sessionResult.reason)
          : null
  }
}

export function nextRefreshMessage({
  afterMutation = false,
  previousMessage = null,
  userInitiated = false,
  warning = null
} = {}) {
  if (warning) {
    return afterMutation ? previousMessage : warning
  }
  return userInitiated ? "Dashboard refreshed." : previousMessage
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
