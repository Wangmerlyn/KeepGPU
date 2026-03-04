import { useEffect, useMemo, useState } from "react"

const defaultForm = {
  gpuIds: "",
  vram: "1GiB",
  interval: "300",
  busyThreshold: "25"
}

async function api(method, path, body) {
  const response = await fetch(path, {
    method,
    headers: {
      "Content-Type": "application/json"
    },
    body: body ? JSON.stringify(body) : undefined
  })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Request failed (${response.status})`)
  }
  return response.json()
}

function parseGpuIds(raw) {
  if (!raw.trim()) {
    return null
  }
  return raw
    .split(",")
    .map((part) => Number(part.trim()))
    .filter((value) => Number.isFinite(value))
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
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState("Awaiting telemetry stream.")

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
      setMessage(`Telemetry warning: ${error.message}`)
    }
  }

  useEffect(() => {
    refresh()
    const timer = window.setInterval(refresh, 2500)
    return () => window.clearInterval(timer)
  }, [])

  async function onStartSession(event) {
    event.preventDefault()
    setBusy(true)
    try {
      const payload = {
        gpu_ids: parseGpuIds(form.gpuIds),
        vram: form.vram,
        interval: Number(form.interval),
        busy_threshold: Number(form.busyThreshold)
      }
      const result = await api("POST", "/api/sessions", payload)
      setMessage(`Session armed: ${result.job_id}`)
      setForm(defaultForm)
      await refresh()
    } catch (error) {
      setMessage(`Start failed: ${error.message}`)
    } finally {
      setBusy(false)
    }
  }

  async function stopSession(jobId) {
    setBusy(true)
    try {
      await api("DELETE", `/api/sessions/${jobId}`)
      setMessage(`Session released: ${jobId}`)
      await refresh()
    } catch (error) {
      setMessage(`Stop failed: ${error.message}`)
    } finally {
      setBusy(false)
    }
  }

  async function stopAllSessions() {
    setBusy(true)
    try {
      await api("DELETE", "/api/sessions")
      setMessage("All sessions released.")
      await refresh()
    } catch (error) {
      setMessage(`Stop-all failed: ${error.message}`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="deck">
      <div className="grid-noise" aria-hidden="true" />
      <header className="masthead glass">
        <p className="eyebrow">KeepGPU / Local Control Plane</p>
        <h1>GPU Keepalive Command Deck</h1>
        <p>
          Fire non-blocking keep sessions, track thermal pulse, and release cards
          without losing terminal control.
        </p>
      </header>

      <section className="stats-row">
        <article className="stat-card glass rise-delay-1">
          <h2>Detected GPUs</h2>
          <p>{counts.gpuCount}</p>
        </article>
        <article className="stat-card glass rise-delay-2">
          <h2>Active Sessions</h2>
          <p>{counts.activeCount}</p>
        </article>
        <article className="stat-card glass rise-delay-3">
          <h2>Avg Utilization</h2>
          <p>{counts.avgUtil === null ? "n/a" : `${counts.avgUtil}%`}</p>
        </article>
      </section>

      <main className="panel-grid">
        <section className="glass panel launch-panel">
          <div className="panel-heading">
            <h2>Launch Keep Session</h2>
            <span className="chip">non-blocking</span>
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
              <span>Interval (sec)</span>
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
            <button className="primary" type="submit" disabled={busy}>
              Arm Session
            </button>
            <button
              className="ghost"
              type="button"
              disabled={busy || sessions.length === 0}
              onClick={stopAllSessions}
            >
              Release All
            </button>
          </form>
        </section>

        <section className="glass panel">
          <div className="panel-heading">
            <h2>Active Sessions</h2>
            <span className="chip">{sessions.length}</span>
          </div>
          <div className="session-list">
            {sessions.length === 0 ? (
              <p className="empty">No active keep sessions.</p>
            ) : (
              sessions.map((session) => (
                <article key={session.job_id} className="session-row">
                  <div>
                    <h3>{session.job_id}</h3>
                    <p>
                      GPUs {JSON.stringify(session.params.gpu_ids)} / {session.params.vram} /
                      {" "}
                      {session.params.interval}s / threshold {session.params.busy_threshold}%
                    </p>
                  </div>
                  <button
                    type="button"
                    className="danger"
                    disabled={busy}
                    onClick={() => stopSession(session.job_id)}
                  >
                    Release
                  </button>
                </article>
              ))
            )}
          </div>
        </section>

        <section className="glass panel span-all">
          <div className="panel-heading">
            <h2>GPU Telemetry</h2>
            <span className="chip">refresh 2.5s</span>
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

      <footer className="status-line">
        <span className="blink" aria-hidden="true" />
        {message}
      </footer>
    </div>
  )
}
