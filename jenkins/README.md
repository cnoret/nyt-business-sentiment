# Jenkins CI server

A local Jenkins (Blue Ocean) with a Docker-in-Docker sidecar, used to run the
project's `Jenkinsfile` (build image, lint, unit tests, JUnit report, optional
Slack notification).

```bash
cd jenkins
docker compose up -d --build
docker logs jenkins-blueocean 2>&1 | grep -A2 "initialAdminPassword"   # first login
```

Open <http://localhost:8090>, install the suggested plugins, then create a
**Pipeline** job pointing at this repository (script path: `Jenkinsfile`).
To get Slack notifications, define a `SLACK_WEBHOOK` environment variable on
the job or globally.

GitHub Actions runs the same checks on every push (see `.github/workflows/`),
so Jenkins is optional and shown here as an alternative CI setup.
