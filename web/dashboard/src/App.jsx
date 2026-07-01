import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { requestJson as api } from "./lib/api"
import {
  AUTO_REFRESH_INTERVAL_MS,
  canRunAutoRefresh,
  canReuseInFlightRefresh,
  fetchDashboardPayloads,
  formatRefreshWarningMessage,
  formatRefreshMode,
  nextRefreshMessage
} from "./lib/refresh"

import {
  buildSessionPayload,
  formatBusyThresholdLabel,
  formatGpuIdentity,
  formatSessionState,
  formatSessionStateDetail,
  formatStopResultMessage,
  formatUtilizationLabel,
  formatUtilizationWidth,
  getRenderableGpus,
  hasReleasableSessions,
  isSessionStopping,
  summarizeDashboardStats
} from "./lib/session"

const defaultForm = {
  gpuIds: "",
  vram: "1GiB",
  interval: "300",
  busyThreshold: "25"
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
  return ids.join(", ")
}

function statusTone(utilization) {
  if (typeof utilization !== "number" || !Number.isFinite(utilization)) {
    return "text-slate-500"
  }
  if (utilization >= 75) {
    return "text-rose-400"
  }
  if (utilization >= 40) {
    return "text-amber-300"
  }
  return "text-emerald-300"
}

function getVisibilityState() {
  if (typeof document === "undefined") {
    return "visible"
  }
  return document.visibilityState
}

export default function App() {
  const [gpus, setGpus] = useState([])
  const [sessions, setSessions] = useState([])
  const [form, setForm] = useState(defaultForm)
  const [startingSession, setStartingSession] = useState(false)
  const [stoppingAll, setStoppingAll] = useState(false)
  const [stoppingIds, setStoppingIds] = useState(() => new Set())
  const [refreshing, setRefreshing] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [visibilityState, setVisibilityState] = useState(getVisibilityState)
  const [message, setMessage] = useState("Connected to KeepGPU service.")
  const refreshPromiseRef = useRef(null)

  const serviceUrl = window.location.origin

  const renderableGpus = useMemo(() => getRenderableGpus(gpus), [gpus])
  const stats = useMemo(
    () => summarizeDashboardStats(renderableGpus, sessions),
    [renderableGpus, sessions]
  )
  const canReleaseAny = hasReleasableSessions(sessions, stoppingIds, stoppingAll)
  const refreshMode = formatRefreshMode(autoRefresh, visibilityState)

  const refresh = useCallback(async ({
    userInitiated = false,
    afterMutation = false,
    previousMessage = null
  } = {}) => {
    if (canReuseInFlightRefresh(refreshPromiseRef.current, afterMutation)) {
      return refreshPromiseRef.current
    }
    if (refreshPromiseRef.current) {
      await refreshPromiseRef.current
      if (refreshPromiseRef.current) {
        return refreshPromiseRef.current
      }
    }

    const refreshPromise = (async () => {
      setRefreshing(true)
      try {
        const result = await fetchDashboardPayloads(api)
        if (result.gpus !== null) {
          setGpus(result.gpus)
        }
        if (result.sessions !== null) {
          setSessions(result.sessions)
        }
        const nextMessage = nextRefreshMessage({
          afterMutation,
          previousMessage,
          userInitiated,
          warning: result.warning
        })
        if (nextMessage) {
          setMessage(nextMessage)
        }
      } catch (error) {
        const nextMessage = nextRefreshMessage({
          afterMutation,
          previousMessage,
          userInitiated,
          warning: formatRefreshWarningMessage(error)
        })
        if (nextMessage) {
          setMessage(nextMessage)
        }
      } finally {
        setRefreshing(false)
        refreshPromiseRef.current = null
      }
    })()

    refreshPromiseRef.current = refreshPromise
    return refreshPromise
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  useEffect(() => {
    if (typeof document === "undefined") {
      return undefined
    }

    function updateVisibilityState() {
      setVisibilityState(document.visibilityState)
    }

    updateVisibilityState()
    document.addEventListener("visibilitychange", updateVisibilityState)
    return () => {
      document.removeEventListener("visibilitychange", updateVisibilityState)
    }
  }, [])

  useEffect(() => {
    if (!canRunAutoRefresh(autoRefresh, visibilityState)) {
      return undefined
    }

    const timer = window.setInterval(refresh, AUTO_REFRESH_INTERVAL_MS)
    return () => window.clearInterval(timer)
  }, [autoRefresh, visibilityState, refresh])

  async function onStartSession(event) {
    event.preventDefault()
    setStartingSession(true)

    try {
      const payload = buildSessionPayload(form)
      const result = await api("POST", "/api/sessions", payload)
      const successMessage = `Session started: ${result.job_id}`
      setForm(defaultForm)
      setMessage(successMessage)
      await refresh({ afterMutation: true, previousMessage: successMessage })
    } catch (error) {
      setMessage(`Start failed: ${error.message}`)
    } finally {
      setStartingSession(false)
    }
  }

  async function stopSession(jobId) {
    setStoppingIds((previous) => {
      const next = new Set(previous)
      next.add(jobId)
      return next
    })

    try {
      const result = await api("DELETE", `/api/sessions/${jobId}`)
      const successMessage = formatStopResultMessage(result)
      setMessage(successMessage)
      await refresh({ afterMutation: true, previousMessage: successMessage })
    } catch (error) {
      setMessage(`Release failed (${jobId}): ${error.message}`)
    } finally {
      setStoppingIds((previous) => {
        const next = new Set(previous)
        next.delete(jobId)
        return next
      })
    }
  }

  async function stopAllSessions() {
    setStoppingAll(true)

    try {
      const result = await api("DELETE", "/api/sessions")
      const successMessage = formatStopResultMessage(result)
      setMessage(successMessage)
      await refresh({ afterMutation: true, previousMessage: successMessage })
    } catch (error) {
      setMessage(`Stop-all failed: ${error.message}`)
    } finally {
      setStoppingAll(false)
    }
  }

  return (
    <div className="min-h-screen bg-shell text-shell-100">
      <div className="mx-auto w-full max-w-7xl px-4 pb-6 pt-8 md:px-6 lg:px-8">
        <header className="mb-6 rounded-2xl border border-white/10 bg-panel px-6 py-5 shadow-soft">
          <p className="mb-2 font-mono text-xs uppercase tracking-[0.16em] text-shell-500">
            KeepGPU Service Console
          </p>
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <h1 className="font-serif text-3xl font-semibold text-shell-50 md:text-4xl">
                Keepalive Dashboard
              </h1>
              <p className="mt-2 max-w-2xl text-sm leading-relaxed text-shell-400 md:text-base">
                A non-blocking control surface for GPU reservation workflows. Start
                sessions, inspect pressure, and release workloads without leaving your
                terminal pipeline.
              </p>
            </div>
            <div className="rounded-xl border border-white/10 bg-shell-900/70 px-4 py-3 text-xs text-shell-300 md:text-sm">
              <p>
                Service: <span className="font-mono text-shell-100">{serviceUrl}</span>
              </p>
              <p className="mt-1">
                Stop daemon: <span className="font-mono text-shell-100">keep-gpu service-stop</span>
              </p>
            </div>
          </div>
        </header>

        <section className="mb-6 grid gap-3 md:grid-cols-3">
          <article className="rounded-xl border border-white/10 bg-panel px-4 py-4 shadow-soft">
            <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-shell-500">
              Detected GPUs
            </p>
            <p className="mt-3 text-3xl font-semibold text-shell-50">{stats.gpuCount}</p>
          </article>
          <article className="rounded-xl border border-white/10 bg-panel px-4 py-4 shadow-soft">
            <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-shell-500">
              Tracked Sessions
            </p>
            <p className="mt-3 text-3xl font-semibold text-shell-50">
              {stats.trackedCount}
            </p>
          </article>
          <article className="rounded-xl border border-white/10 bg-panel px-4 py-4 shadow-soft">
            <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-shell-500">
              Average Utilization
            </p>
            <p className="mt-3 text-3xl font-semibold text-shell-50">
              {stats.averageUtilization === null ? "n/a" : `${stats.averageUtilization}%`}
            </p>
          </article>
        </section>

        <main className="grid gap-4 lg:grid-cols-12">
          <section className="rounded-2xl border border-white/10 bg-panel p-5 shadow-soft lg:col-span-5">
            <h2 className="font-serif text-xl font-medium text-shell-50">Start Session</h2>
            <p className="mt-1 text-sm text-shell-400">
              Use the GPU number shown on telemetry cards. Leave blank for all visible GPUs.
            </p>

            <form
              className="mt-5 grid grid-cols-1 gap-3 md:grid-cols-2"
              noValidate
              onSubmit={onStartSession}
            >
              <label className="field-label md:col-span-2">
                <span>GPU IDs (visible ordinals)</span>
                <input
                  className="field-input"
                  value={form.gpuIds}
                  onChange={(event) =>
                    setForm((previous) => ({ ...previous, gpuIds: event.target.value }))
                  }
                  placeholder="0,1"
                />
              </label>

              <label className="field-label">
                <span>VRAM</span>
                <input
                  className="field-input"
                  value={form.vram}
                  onChange={(event) =>
                    setForm((previous) => ({ ...previous, vram: event.target.value }))
                  }
                  placeholder="1GiB"
                />
              </label>

              <label className="field-label">
                <span>Interval (sec)</span>
                <input
                  className="field-input"
                  type="number"
                  min="0.001"
                  step="any"
                  value={form.interval}
                  onChange={(event) =>
                    setForm((previous) => ({ ...previous, interval: event.target.value }))
                  }
                />
              </label>

              <label className="field-label md:col-span-2">
                <span>Busy threshold (%)</span>
                <input
                  className="field-input"
                  type="number"
                  min="-1"
                  max="100"
                  value={form.busyThreshold}
                  onChange={(event) =>
                    setForm((previous) => ({
                      ...previous,
                      busyThreshold: event.target.value
                    }))
                  }
                />
              </label>

              <button
                type="submit"
                disabled={startingSession || stoppingAll}
                className="btn-primary"
              >
                {startingSession ? "Starting..." : "Start Keepalive"}
              </button>

              <button
                type="button"
                disabled={!canReleaseAny}
                className="btn-muted"
                onClick={stopAllSessions}
              >
                {stoppingAll ? "Releasing..." : "Release All"}
              </button>
            </form>
          </section>

          <section className="rounded-2xl border border-white/10 bg-panel p-5 shadow-soft lg:col-span-7">
            <div className="flex items-center justify-between">
              <h2 className="font-serif text-xl font-medium text-shell-50">Tracked Sessions</h2>
              <span className="rounded-full border border-white/10 px-3 py-1 font-mono text-xs text-shell-400">
                {sessions.length} tracked
              </span>
            </div>

            <div className="mt-4 space-y-2">
              {sessions.length === 0 ? (
                <p className="rounded-xl border border-dashed border-white/10 px-4 py-6 text-sm text-shell-500">
                  No active keepalive sessions.
                </p>
              ) : (
                sessions.map((session) => {
                  const currentlyStopping = isSessionStopping(session, stoppingIds, stoppingAll)
                  const stateLabel = formatSessionState(session)
                  const stateDetail = formatSessionStateDetail(session)

                  return (
                    <article
                      key={session.job_id}
                      className="flex flex-col gap-3 rounded-xl border border-white/10 bg-shell-900/60 p-4 md:flex-row md:items-center md:justify-between"
                    >
                      <div>
                        <div className="flex flex-wrap items-center gap-2">
                          <h3 className="font-mono text-sm text-shell-100">
                            {session.job_id}
                          </h3>
                          <span className="rounded-full border border-white/10 px-2 py-0.5 font-mono text-[11px] uppercase tracking-[0.1em] text-shell-400">
                            {stateLabel}
                          </span>
                        </div>
                        <p className="mt-1 text-sm text-shell-400">
                          Visible GPUs {formatGpuTarget(session.params.gpu_ids)} ·{" "}
                          {session.params.vram} · {session.params.interval}s · threshold{" "}
                          {formatBusyThresholdLabel(session.params.busy_threshold)}
                        </p>
                        {stateDetail ? (
                          <p className="mt-2 max-w-xl text-xs leading-relaxed text-shell-500">
                            {stateDetail}
                          </p>
                        ) : null}
                      </div>
                      <button
                        type="button"
                        disabled={currentlyStopping}
                        onClick={() => stopSession(session.job_id)}
                        className="btn-danger md:min-w-28"
                      >
                        {currentlyStopping ? "Releasing..." : "Release"}
                      </button>
                    </article>
                  )
                })
              )}
            </div>
          </section>

          <section className="rounded-2xl border border-white/10 bg-panel p-5 shadow-soft lg:col-span-12">
            <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <h2 className="font-serif text-xl font-medium text-shell-50">GPU Telemetry</h2>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  className="btn-muted"
                  disabled={refreshing}
                  onClick={() => refresh({ userInitiated: true })}
                >
                  {refreshing ? "Refreshing..." : "Refresh Now"}
                </button>
                <label className="inline-flex items-center gap-2 rounded-full border border-white/10 px-3 py-1.5 text-xs text-shell-300">
                  <input
                    type="checkbox"
                    aria-label="Auto refresh"
                    checked={autoRefresh}
                    onChange={(event) => setAutoRefresh(event.target.checked)}
                  />
                  <span>Auto refresh</span>
                </label>
                <span className="font-mono text-xs uppercase tracking-[0.1em] text-shell-500">
                  {refreshMode}
                </span>
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              {renderableGpus.length === 0 ? (
                <p className="col-span-full rounded-xl border border-dashed border-white/10 px-4 py-6 text-sm text-shell-500">
                  No GPU telemetry available.
                </p>
              ) : (
                renderableGpus.map((gpu) => {
                  const utilizationBarWidth = formatUtilizationWidth(gpu.utilization)

                  return (
                    <article
                      key={`${gpu.platform}-${gpu.id}`}
                      className="rounded-xl border border-white/10 bg-shell-900/65 p-4"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <h3 className="text-sm font-medium text-shell-100">
                          {gpu.name}
                          <small className="mt-1 block font-mono text-[11px] text-shell-500">
                            {formatGpuIdentity(gpu)} · {gpu.platform}
                          </small>
                        </h3>
                        <span className={`font-mono text-xs ${statusTone(gpu.utilization)}`}>
                          {formatUtilizationLabel(gpu.utilization)}
                        </span>
                      </div>

                      <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-shell-800">
                        {utilizationBarWidth === null ? null : (
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-emerald-300 via-amber-300 to-rose-300"
                            style={{ width: utilizationBarWidth }}
                          />
                        )}
                      </div>

                      <p className="mt-3 text-sm text-shell-400">
                        {formatBytes(gpu.memory_used)} / {formatBytes(gpu.memory_total)} used
                      </p>
                    </article>
                  )
                })
              )}
            </div>
          </section>
        </main>

        <footer className="mt-4 rounded-xl border border-white/10 bg-panel px-4 py-3 font-mono text-xs text-shell-400">
          {message}
        </footer>
      </div>
    </div>
  )
}
