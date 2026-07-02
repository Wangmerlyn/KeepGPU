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

function readRefreshList(result, fieldName, malformedMessage) {
  if (result.status === "rejected") {
    return {
      items: null,
      warning: formatRefreshWarningMessage(result.reason)
    }
  }
  const items = result.value?.[fieldName]
  if (!Array.isArray(items)) {
    return {
      items: null,
      warning: formatRefreshWarningMessage(new Error(malformedMessage))
    }
  }
  return { items, warning: null }
}

export async function fetchDashboardPayloads(requestJson) {
  const [gpuResult, sessionResult] = await Promise.allSettled([
    requestJson("GET", "/api/gpus"),
    requestJson("GET", "/api/sessions")
  ])
  const gpuPayload = readRefreshList(
    gpuResult,
    "gpus",
    "malformed GPU list response"
  )
  const sessionPayload = readRefreshList(
    sessionResult,
    "active_jobs",
    "malformed session list response"
  )

  return {
    gpus: gpuPayload.items,
    sessions: sessionPayload.items,
    warning: gpuPayload.warning ?? sessionPayload.warning
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
