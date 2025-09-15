# Lando Full Deploy Bundle

This bundle includes Phase II CoreNode Lando implementation, tests, GitHub Actions CI, and a Helm chart for Kubernetes deployment.

## Contents
- app/: Flask server and model module (LandoNet)
- core/: CoreNode meta-learner and trainer logic
- data/: example_pairs.json (small dataset)
- tests/: pytest tests
- helm/lando/: Helm chart (Deployment, Service, Ingress)
- .github/workflows/: CI and Deploy workflows
- docker-compose.yml for local testing
- README.md (this file)

## Quick local run (Docker Compose)
1. Build & run:
   ```bash
   docker-compose up --build
   ```
2. Health check:
   ```bash
   curl http://localhost:8080/healthz
   ```
3. Train (locally in container or host):
   ```bash
   # inside container or local python environment
   python -m core.trainer --epochs 5
   ```

## Deploy to Kubernetes (recommended via GitHub Actions)
1. Push this repo to GitHub.
2. Set secret `KUBECONFIG_DATA` in repo settings: base64-encode your kubeconfig and paste value.
   ```bash
   cat ~/.kube/config | base64 | tr -d '\n'
   ```
3. The `deploy.yml` workflow builds image and pushes to ghcr.io and runs `helm upgrade --install`.
   - Ensure the repository path in `helm/lando/values.yaml` is updated (replace YOUR-ORG/YOUR-REPO).
   - GitHub Actions uses `GITHUB_TOKEN` to authenticate to GHCR for pushing images.
4. After workflow completes, verify:
   ```bash
   kubectl get pods -n lando
   kubectl port-forward svc/lando 8080:8080 -n lando
   curl http://localhost:8080/healthz
   ```

## Manual image push (optional)
1. Build locally and push to GHCR:
   ```bash
   IMAGE=ghcr.io/<OWNER>/<REPO>/lando:latest
   docker build -t $IMAGE ./app
   echo "<PAT>" | docker login ghcr.io -u <OWNER> --password-stdin
   docker push $IMAGE
   ```
2. Deploy with Helm:
   ```bash
   helm upgrade --install lando helm/lando --namespace lando --create-namespace --set image.repository=ghcr.io/<OWNER>/<REPO>/lando --set image.tag=latest
   ```

## Notes
- The Helm chart uses readiness/liveness probes on `/healthz`.
- The CoreNode trainer can grow/prune model width; it saves models to `models/lando_phase2.pt`.
- CI runs `pytest` and builds the container to validate startup.

