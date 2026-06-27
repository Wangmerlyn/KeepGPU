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
  if (!Number.isInteger(parsed) || parsed < -1) {
    throw new Error("Busy threshold must be an integer >= -1")
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

export function isSessionStopping(jobId, stoppingIds, stoppingAll) {
  return stoppingAll || stoppingIds.has(jobId)
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
