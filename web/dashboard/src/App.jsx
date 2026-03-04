import { useEffect, useMemo, useState } from "react"

const defaultForm = {
  gpuIds: "",
  vram: "1GiB",
  interval: "300",
  busyThreshold: "25"
}

const REQUEST_TIMEOUT_MS = 10000

async function api(method, path, body) {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)
  try {
    const response = await fetch(path, {
      method,
      headers: {
        "Content-Type": "application/json"
      },
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal
    })
    if (!response.ok) {
      const text = await response.text()
      throw new Error(text || `Request failed (${response.status})`)
    }
    return response.json()
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("Request timed out")
    }
    throw error
  } finally {
    window.clearTimeout(timeout)
  }
}

function parseGpuIds(raw) {
  const value = raw.trim()
  if (!value) {
    return null
  }

  const parts = value.split(",").map((part) => part.trim())
  if (parts.some((part) => !/^\d+$/.test(part))) {
    throw new Error("GPU IDs must be comma-separated integers, for example: 0,1")
  }
  return parts.map((part) => Number(part))
}

function formatBytes(value) {
  if (value === null || value === undefined) {
    return "n/a"
  }
  const units = ["B", "KB", "MB", "GB", "TB"]
  let current = Number(value)
  let index = 0
  while (current >= 1024 && index < units.length - 1) {
    current /= 1024
    index += 1
  }
  return `${current.toFixed(current >= 10 ? 0 : 1)} ${units[index]}`
}

function formatGpuTarget(ids) {
  if (!ids || ids.length === 0) {
    return "all visible"
  }
  return ids.join(",")
}

function utilizationTone(util) {
  if (util === null || util === undefined) {
    return "muted"
  }
  if (util >= 75) {
    return "alert"
  }
  if (util >= 40) {
    return "warm"
  }
  return "cool"
}

export default function App() {
  const [gpus, setGpus] = useState([])
  const [sessions, setSessions] = useState([])
  const [form, setForm] = useState(defaultForm)
  const [startingSession, setStartingSession] = useState(false)
  const [stoppingAll, setStoppingAll] = useState(false)
  const [stoppingIds, setStoppingIds] = useState(() => new Set())
  const [message, setMessage] = useState("Connected to KeepGPU service.")

  const serviceUrl = window.location.origin

  const counts = useMemo(() => {
    const gpuCount = gpus.length
    const activeCount = sessions.length
    const avgUtil =
      gpus.length === 0
        ? null
        : Math.round(
            gpus.reduce((acc, gpu) => acc + (gpu.utilization ?? 0), 0) / gpus.length
          )
    return { gpuCount, activeCount, avgUtil }
  }, [gpus, sessions])

  async function refresh() {
    try {
      const [gpuPayload, sessionPayload] = await Promise.all([
        api("GET", "/api/gpus"),
        api("GET", "/api/sessions")
      ])
      setGpus(gpuPayload.gpus ?? [])
      setSessions(sessionPayload.active_jobs ?? [])
    } catch (error) {
      setMessage(`Refresh warning: ${error.message}`)
    }
  }

  useEffect(() => {
    refresh()
    const timer = window.setInterval(refresh, 3000)
    return () => window.clearInterval(timer)
  }, [])

  async function onStartSession(event) {
    event.preventDefault()
    setStartingSession(true)
    try {
      const payload = {
        gpu_ids: parseGpuIds(form.gpuIds),
        vram: form.vram,
        interval: Number(form.interval),
        busy_threshold: Number(form.busyThreshold)
      }
      const result = await api("POST", "/api/sessions", payload)
      setMessage(`Session started: ${result.job_id}`)
      setForm(defaultForm)
      await refresh()
    } catch (error) {
      setMessage(`Start failed: ${error.message}`)
    } finally {
      setStartingSession(false)
    }
  }

  async function stopSession(jobId) {
    setStoppingIds((prev) => {
      const next = new Set(prev)
      next.add(jobId)
      return next
    })
    try {
      await api("DELETE", `/api/sessions/${jobId}`)
      setMessage(`Session released: ${jobId}`)
      await refresh()
    } catch (error) {
      setMessage(`Release failed (${jobId}): ${error.message}`)
    } finally {
      setStoppingIds((prev) => {
        const next = new Set(prev)
        next.delete(jobId)
        return next
      })
    }
  }

  async function stopAllSessions() {
    setStoppingAll(true)
    try {
      await api("DELETE", "/api/sessions")
      setMessage("All sessions released.")
      await refresh()
    } catch (error) {
      setMessage(`Stop-all failed: ${error.message}`)
    } finally {
      setStoppingAll(false)
    }
  }

  return (
    <div className="deck">
      <header className="masthead panel">
        <p className="eyebrow">KeepGPU Operations</p>
        <h1>GPU Keepalive Console</h1>
        <p>
          Manage non-blocking keepalive sessions with a clean control surface.
          Start sessions, monitor pressure, and release devices without blocking
          your terminal pipeline.
        </p>
        <p className="service-hint">
          Service endpoint <code>{serviceUrl}</code> · daemon command
          <code>keep-gpu service-stop</code>
        </p>
      </header>

      <section className="stats-row">
        <article className="stat-card panel">
          <h2>Detected GPUs</h2>
          <p>{counts.gpuCount}</p>
        </article>
        <article className="stat-card panel">
          <h2>Active Sessions</h2>
          <p>{counts.activeCount}</p>
        </article>
        <article className="stat-card panel">
          <h2>Average Utilization</h2>
          <p>{counts.avgUtil === null ? "n/a" : `${counts.avgUtil}%`}</p>
        </article>
      </section>

      <main className="panel-grid">
        <section className="panel">
          <div className="panel-heading">
            <h2>Start Session</h2>
          </div>
          <form onSubmit={onStartSession} className="form-grid">
            <label>
              <span>GPU IDs</span>
              <input
                value={form.gpuIds}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, gpuIds: event.target.value }))
                }
                placeholder="0,1"
              />
            </label>
            <label>
              <span>VRAM</span>
              <input
                value={form.vram}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, vram: event.target.value }))
                }
                placeholder="1GiB"
              />
            </label>
            <label>
              <span>Interval (seconds)</span>
              <input
                type="number"
                min="1"
                value={form.interval}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, interval: event.target.value }))
                }
              />
            </label>
            <label>
              <span>Busy Threshold (%)</span>
              <input
                type="number"
                min="-1"
                value={form.busyThreshold}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, busyThreshold: event.target.value }))
                }
              />
            </label>
            <button className="primary" type="submit" disabled={startingSession || stoppingAll}>
              {startingSession ? "Starting..." : "Start Keepalive"}
            </button>
            <button
              className="ghost"
              type="button"
              disabled={stoppingAll || sessions.length === 0}
              onClick={stopAllSessions}
            >
              {stoppingAll ? "Releasing..." : "Release All"}
            </button>
          </form>
        </section>

        <section className="panel">
          <div className="panel-heading">
            <h2>Active Sessions</h2>
          </div>
          <div className="session-list">
            {sessions.length === 0 ? (
              <p className="empty">No active keepalive sessions.</p>
            ) : (
              sessions.map((session) => {
                const isStopping = stoppingIds.has(session.job_id) || stoppingAll
                return (
                  <article key={session.job_id} className="session-row">
                    <div>
                      <h3>{session.job_id}</h3>
                      <p>
                        GPUs {formatGpuTarget(session.params.gpu_ids)} / {session.params.vram}
                        / {session.params.interval}s / threshold {session.params.busy_threshold}%
                      </p>
                    </div>
                    <button
                      type="button"
                      className="danger"
                      disabled={isStopping}
                      onClick={() => stopSession(session.job_id)}
                    >
                      {isStopping ? "Releasing..." : "Release"}
                    </button>
                  </article>
                )
              })
            )}
          </div>
        </section>

        <section className="panel span-all">
          <div className="panel-heading">
            <h2>GPU Telemetry</h2>
            <span className="refresh-tag">refresh 3s</span>
          </div>
          <div className="telemetry-grid">
            {gpus.length === 0 ? (
              <p className="empty">No GPU telemetry available.</p>
            ) : (
              gpus.map((gpu) => (
                <article key={`${gpu.platform}-${gpu.id}`} className="telemetry-card">
                  <header>
                    <h3>
                      {gpu.name}
                      <small>
                        {gpu.platform}:{gpu.id}
                      </small>
                    </h3>
                    <span className={`util-pill ${utilizationTone(gpu.utilization)}`}>
                      {gpu.utilization ?? "n/a"}%
                    </span>
                  </header>
                  <div className="meter">
                    <div
                      className="meter-fill"
                      style={{ width: `${Math.max(0, Math.min(100, gpu.utilization ?? 0))}%` }}
                    />
                  </div>
                  <p>
                    {formatBytes(gpu.memory_used)} / {formatBytes(gpu.memory_total)} used
                  </p>
                </article>
              ))
            )}
          </div>
        </section>
      </main>

      <footer className="status-line">{message}</footer>
    </div>
  )
}
