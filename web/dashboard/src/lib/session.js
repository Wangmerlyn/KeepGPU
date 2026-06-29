const INTEGER_PATTERN = /^\d+$/

export function parseGpuIds(raw) {
  const value = raw.trim()
  if (!value) {
    return null
  }

  const parts = value.split(",").map((part) => part.trim())
  if (parts.some((part) => !INTEGER_PATTERN.test(part))) {
    throw new Error(
      "GPU IDs must be comma-separated visible ordinals, for example: 0,1"
    )
  }

  return parts.map((part) => Number(part))
}

export function parsePositiveNumber(value, fieldName) {
  if (typeof value === "boolean") {
    throw new Error(`${fieldName} must be finite and positive`)
  }
  const normalized = typeof value === "string" ? value.trim() : value
  if (normalized === "") {
    throw new Error(`${fieldName} must be finite and positive`)
  }
  const parsed = Number(normalized)
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${fieldName} must be finite and positive`)
  }
  return parsed
}

export function parseBusyThreshold(value) {
  if (typeof value === "boolean") {
    throw new Error("Busy threshold must be -1 or an integer between 0 and 100")
  }
  const normalized = typeof value === "string" ? value.trim() : value
  if (normalized === "") {
    throw new Error("Busy threshold must be -1 or an integer between 0 and 100")
  }
  const parsed = Number(normalized)
  if (!Number.isInteger(parsed) || (parsed !== -1 && (parsed < 0 || parsed > 100))) {
    throw new Error("Busy threshold must be -1 or an integer between 0 and 100")
  }
  return parsed
}

export function buildSessionPayload(form) {
  const trimmedVram = form.vram.trim()
  if (!trimmedVram) {
    throw new Error("VRAM value is required")
  }

  return {
    gpu_ids: parseGpuIds(form.gpuIds),
    vram: trimmedVram,
    interval: parsePositiveNumber(form.interval, "Interval"),
    busy_threshold: parseBusyThreshold(form.busyThreshold)
  }
}

export function formatGpuIdentity(gpu) {
  const visibleId = gpu?.visible_id ?? gpu?.id
  const label =
    visibleId === undefined || visibleId === null ? "GPU n/a" : `GPU ${visibleId}`
  if (
    gpu?.physical_id !== undefined &&
    gpu.physical_id !== null &&
    gpu.physical_id !== visibleId
  ) {
    return `${label} (physical ${gpu.physical_id})`
  }
  return label
}

function isKnownUtilization(value) {
  return typeof value === "number" && Number.isFinite(value)
}

export function getRenderableGpus(gpus = []) {
  return gpus.filter((gpu) => gpu !== null && gpu !== undefined)
}

export function summarizeDashboardStats(gpus = [], sessions = []) {
  const renderableGpus = getRenderableGpus(gpus)
  const knownUtilizations = renderableGpus
    .map((gpu) => gpu?.utilization)
    .filter(isKnownUtilization)
  const averageUtilization =
    knownUtilizations.length === 0
      ? null
      : Math.round(
          knownUtilizations.reduce((acc, utilization) => acc + utilization, 0) /
            knownUtilizations.length
        )

  return {
    gpuCount: renderableGpus.length,
    trackedCount: sessions.length,
    averageUtilization
  }
}

export function formatUtilizationLabel(utilization) {
  return isKnownUtilization(utilization) ? `${utilization}%` : "n/a"
}

export function formatUtilizationWidth(utilization) {
  if (!isKnownUtilization(utilization)) {
    return null
  }
  return `${Math.max(0, Math.min(100, utilization))}%`
}

export function formatBusyThresholdLabel(threshold) {
  if (
    threshold === null ||
    threshold === undefined ||
    threshold === "" ||
    typeof threshold === "boolean"
  ) {
    return "n/a"
  }

  const numericThreshold = Number(threshold)
  if (!Number.isFinite(numericThreshold)) {
    return "n/a"
  }

  return numericThreshold === -1 ? "unconditional" : `${numericThreshold}%`
}

export function isSessionStopping(sessionOrJobId, stoppingIds, stoppingAll) {
  const isSession = typeof sessionOrJobId === "object" && sessionOrJobId !== null
  const jobId = isSession ? sessionOrJobId.job_id : sessionOrJobId
  return stoppingAll || sessionOrJobId?.state === "stopping" || stoppingIds.has(jobId)
}

export function hasReleasableSessions(sessions, stoppingIds, stoppingAll) {
  return (
    !stoppingAll &&
    sessions.some((session) => !isSessionStopping(session, stoppingIds, false))
  )
}

export function formatSessionState(session) {
  switch (session?.state) {
    case undefined:
    case null:
    case "active":
      return "Active"
    case "starting":
      return "Starting"
    case "stopping":
      return "Releasing"
    case "runtime_failed":
      return "Runtime failed"
    case "stop_failed":
      return "Release failed"
    default:
      return String(session.state)
  }
}

export function formatSessionStateDetail(session) {
  if (session?.state === "starting") {
    return "Controller startup is still in progress."
  }
  if (session?.state === "stopping") {
    return session.last_error || "Release is still completing in the background."
  }
  if (session?.state === "stop_failed") {
    return session.last_error || "Release failed. Inspect the session before retrying."
  }
  if (session?.state === "runtime_failed") {
    return (
      session.last_error ||
      "Keep worker failed after startup. Inspect the session before retrying."
    )
  }
  return null
}

function asArray(value) {
  return Array.isArray(value) ? value : []
}

function sessionLabel(count) {
  return count === 1 ? "session" : "sessions"
}

function formatIds(ids) {
  return ids.join(", ")
}

function formatErrors(errors) {
  if (Array.isArray(errors)) {
    return errors.map((error) => String(error))
  }

  if (errors && typeof errors === "object") {
    return Object.entries(errors).map(([jobId, error]) => `${jobId}: ${String(error)}`)
  }

  return []
}

export function formatStopResultMessage(result) {
  const stopped = asArray(result?.stopped)
  const timedOut = asArray(result?.timed_out)
  const failed = asArray(result?.failed)
  const errors = formatErrors(result?.errors)
  const parts = []

  if (timedOut.length > 0) {
    parts.push(
      `Timed out stopping ${sessionLabel(timedOut.length)}: ${formatIds(timedOut)}.`
    )
  }

  if (failed.length > 0) {
    parts.push(
      `Failed to release ${sessionLabel(failed.length)}: ${formatIds(failed)}.`
    )
  }

  if (stopped.length > 0) {
    parts.push(`Released ${sessionLabel(stopped.length)}: ${formatIds(stopped)}.`)
  }

  if (errors.length > 0) {
    parts.push(`Errors: ${errors.join("; ")}.`)
  }

  return parts.length > 0 ? parts.join(" ") : "No sessions were released."
}
