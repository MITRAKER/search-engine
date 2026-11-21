# Security Policy

## Supported versions

Only the latest release receives security fixes.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Instead, use
GitHub's private reporting: open the repository's **Security** tab and click
**Report a vulnerability**.

Include what you found, how to reproduce it, and what an attacker could do
with it. You should get a first answer within 7 days.

## How this project stays secure

- CodeQL scans the code and the workflows on every pull request and weekly.
- `pip-audit` checks every dependency against known vulnerabilities in CI,
  every day.
- Dependabot proposes updates for dependencies and GitHub Actions.
- GitHub Actions are pinned to full commit SHAs, so a moved tag cannot change
  what runs.
- Workflows get read-only tokens unless a job needs more.
- Releases carry a signed build provenance attestation.
