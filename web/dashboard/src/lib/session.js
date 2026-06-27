const INTEGER_PATTERN = /^\d+$/

export function parseGpuIds(raw) {
  const value = raw.trim()
  if (!value) {
    return null
  }

  const parts = value.split(",").map((part) => part.trim())
  if (parts.some((part) => !INTEGER_PATTERN.test(part))) {
    throw new Error("GPU IDs must be comma-separated integers, for example: 0,1")
  }

  return parts.map((part) => Number(part))
}

export function parsePositiveInt(value, fieldName) {
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${fieldName} must be an integer >= 1`)
  }
  return parsed
}

export function parseBusyThreshold(value) {
  const parsed = Number(value)
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
    interval: parsePositiveInt(form.interval, "Interval"),
    busy_threshold: parseBusyThreshold(form.busyThreshold)
  }
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
    case "stopping":
      return "Releasing"
    case "stop_failed":
      return "Release failed"
    default:
      return String(session.state)
  }
}

export function formatSessionStateDetail(session) {
  if (session?.state === "stopping") {
    return session.last_error || "Release is still completing in the background."
  }
  if (session?.state === "stop_failed") {
    return session.last_error || "Release failed. Inspect the session before retrying."
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
