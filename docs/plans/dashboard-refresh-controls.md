# Dashboard Refresh Controls

## Background

The dashboard previously called `/api/gpus` and `/api/sessions` every 3 seconds
for as long as the page stayed open. That made the control surface itself a
steady telemetry workload, even when the user only wanted to leave the page
nearby as a manual console.

## Goal

Keep the dashboard useful while matching KeepGPU's low-power posture: one load
on open, manual refresh by default, and optional auto refresh that pauses when
the tab is hidden.

## Solution

- Add a manual **Refresh Now** action.
- Add an explicit **Auto refresh** toggle.
- Use a 10-second auto-refresh interval only when the toggle is enabled and the
  tab is visible.
- Collapse overlapping refresh attempts into one in-flight request.
- Update dashboard docs, AGENTS guidance, tests, and packaged static assets.

## Validation

- `npm --prefix web/dashboard test`
- `npm --prefix web/dashboard run build`
- `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py -q`
- `PYTHONPATH=$PWD/src mkdocs build`
- `pre-commit run --all-files`
