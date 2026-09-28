# Security policy

DATA AUTOPSY runs entirely locally. It makes **no outbound network calls** during
analysis, sends no telemetry, and stores nothing outside your machine.

## Data handling

- Uploaded files are processed in memory by the Streamlit process you started.
- Generated reports are written only where you ask them to be written.
- The only external request the application can trigger is loading Plotly from a
  CDN when you open an **exported HTML report** in a browser. The dashboard itself
  serves everything from the local process.

## Secrets

Never commit credentials. `.gitignore` already excludes `.env`, `.netrc`,
`.git-credentials` and `.streamlit/secrets.toml`.

## Reporting a vulnerability

Open a private security advisory on GitHub (Security → Advisories → Report a
vulnerability) rather than a public issue. Please include a reproduction and the
affected version.
