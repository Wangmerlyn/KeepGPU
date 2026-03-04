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
