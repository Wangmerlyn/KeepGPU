const SERVER_STARTUP_STOP_WAIT_TIMEOUT_MS = 10000
const SERVER_RELEASE_TIMEOUT_MS = 10000
export const REQUEST_TIMEOUT_MS = SERVER_RELEASE_TIMEOUT_MS + 5000
export const STOP_REQUEST_TIMEOUT_MS =
  SERVER_STARTUP_STOP_WAIT_TIMEOUT_MS + SERVER_RELEASE_TIMEOUT_MS + 5000

function timeoutForMethod(method) {
  return String(method).toUpperCase() === "DELETE"
    ? STOP_REQUEST_TIMEOUT_MS
    : REQUEST_TIMEOUT_MS
}

function cleanMessage(value) {
  return typeof value === "string" && value.trim() ? value.trim() : null
}

export function formatApiErrorMessage(responseBody, status) {
  const fallback = cleanMessage(responseBody) || `Request failed (${status})`
  const body = cleanMessage(responseBody)
  if (!body) {
    return fallback
  }

  try {
    const payload = JSON.parse(body)
    const isObject = payload !== null && typeof payload === "object"
    const errorIsObject = isObject && payload.error && typeof payload.error === "object"
    const nestedMessage = errorIsObject ? cleanMessage(payload.error.message) : null
    const topError = isObject ? cleanMessage(payload.error) : null
    const topMessage = isObject ? cleanMessage(payload.message) : null

    if (nestedMessage) {
      return nestedMessage
    }
    if (topError) {
      return topError
    }
    if (topMessage) {
      return topMessage
    }
    if (errorIsObject && "message" in payload.error) {
      return `Request failed (${status})`
    }
    if (isObject && ("error" in payload || "message" in payload)) {
      return `Request failed (${status})`
    }
    return fallback
  } catch {
    return fallback
  }
}

export async function requestJson(method, path, body) {
  const controller = new AbortController()
  const timeout = globalThis.setTimeout(
    () => controller.abort(),
    timeoutForMethod(method)
  )
  const hasBody = arguments.length >= 3

  try {
    const response = await fetch(path, {
      method,
      headers: {
        "Content-Type": "application/json"
      },
      body: hasBody ? JSON.stringify(body) : undefined,
      signal: controller.signal
    })

    if (!response.ok) {
      const text = await response.text()
      throw new Error(formatApiErrorMessage(text, response.status))
    }

    return response.json()
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("Request timed out")
    }
    throw error
  } finally {
    globalThis.clearTimeout(timeout)
  }
}
