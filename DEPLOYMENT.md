# Deployment Guide

---

## 0. What gets deployed

- `app.py`: a Gradio web page (upload an image, see the overlay).
- `checkpoints/synthetic_demo.pt`: the small trained model (about 180 KB).
- A `Dockerfile` that packages everything and listens on port **7860**.

You do **not** need a GPU. The model is tiny and runs on a free CPU host.

---

## 1. Test locally first

```powershell
venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-app.txt -r requirements-dev.txt
python -m pytest -q            # all tests should pass
python app.py                  # open http://localhost:7860 and try the example images
```

Optional, if Docker Desktop is installed (it needs WSL2 on Windows, which is extra setup; skip it if you only deploy to Hugging Face):

```powershell
docker compose up --build      # demo on :7860, API on :8000/docs
```

Do not continue until the local demo works.

---

## 2. Put the code on GitHub

Your repository already exists, so **do not force-push** over it. Work inside a clone.

```powershell
git clone https://github.com/Rachi-collab/Graph-guided-reinforcement-learning-for-medical-image-segmentation.git
cd Graph-guided-reinforcement-learning-for-medical-image-segmentation
```

Copy the new project files into this folder (keep your own files if they are newer; compare before overwriting), then:

```powershell
git status                      # check you are not adding venv/, dataset/ or big files
git add .
git commit -m "Add inference API, Gradio demo, tests, Docker and CI"
git push origin main
```

If `git push` says the remote has changes you do not have: run `git pull --rebase origin main`, resolve any conflicts, then push again.

After the push, open the **Actions** tab on GitHub. The `CI` workflow should turn green (tests, then a Docker build). If it is red, open the failed step and read the log. It is the first real test of the `Dockerfile`.

**Never commit:** patient data, `venv/`, large checkpoints (over about 50 MB; use Git LFS or the Hugging Face Hub instead), or secrets.

---

## 3. Option A (recommended, free): Hugging Face Spaces

Spaces gives you a public URL for free on a CPU machine. A free Space may sleep after a period of inactivity and wake up on the next visit (it takes a minute).

1. Create an account at **huggingface.co**.
2. Click your profile → **New Space**. Choose:
   - Space name: for example `graph-rl-segmentation`
   - SDK: **Docker** (blank template)
   - Hardware: **CPU basic (free)**
   - Visibility: Public
3. Create an access token: profile → **Settings → Access Tokens → New token**, role **Write**. Copy it.
4. Clone the empty Space and copy the project in:

```powershell
git clone https://huggingface.co/spaces/<your-username>/graph-rl-segmentation
# Username = your Hugging Face username. Password = the access token (not your account password).
```

5. Copy these into the Space folder: `gseg/`, `app.py`, `api.py`, `checkpoints/`, `examples/`, `Dockerfile`, `requirements.txt`, `requirements-app.txt`, `.dockerignore`.
6. Copy `deploy/huggingface/SPACE_README.md` into the Space folder and **rename it to `README.md`**. The block at the top (between `---` lines) tells Hugging Face to use Docker and port 7860, so do not delete it.
7. Push:

```powershell
cd graph-rl-segmentation
git add .
git commit -m "Deploy demo"
git push
```

8. Open the Space page. It shows **Building**, then **Running** (the first build takes several minutes because it installs PyTorch). Your public link is `https://huggingface.co/spaces/<your-username>/graph-rl-segmentation`.

If the build fails, open the **Logs** tab. Common causes are listed in section 6.

---

## 4. Option B: Render (Docker web service)

Render builds straight from your GitHub repo and redeploys on every push. The free plan spins the service down when idle, so the first visit after a while is slow.

1. Create an account at **render.com** and connect your GitHub.
2. **New → Web Service** → select the repository.
3. Settings:
   - Runtime / Language: **Docker** (Render finds the `Dockerfile`)
   - Instance type: **Free** (or the cheapest paid tier if the free one runs out of memory)
   - Health check path: `/`
4. Create the service. Render sets the `PORT` variable and `app.py` reads it, so no extra setting is needed.
5. When the deploy finishes, open the `onrender.com` URL.

Optional: deploy the FastAPI service as a second Render service with the start command `uvicorn api:app --host 0.0.0.0 --port $PORT`.

---

## 5. Option C: your own server or a VM

On any Linux machine with Docker:

```bash
git clone <your repo> && cd <repo>
docker compose up -d --build
```

Put a reverse proxy (Caddy or Nginx) in front for HTTPS. Do this only if you are comfortable with server administration.

---

## 6. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Space stuck on "Building" or fails at `pip install` | Check the Logs tab. Make sure `requirements.txt` and `requirements-app.txt` were copied. The CPU PyTorch install in the `Dockerfile` needs internet during build; retry the build. |
| Space shows "Application error" or never becomes Running | The app must listen on port **7860** and `app_port: 7860` must be in the README header. Do not change either. |
| `FileNotFoundError: checkpoints/synthetic_demo.pt` | The `checkpoints/` folder was not copied or was git-ignored. Run `git add -f checkpoints/synthetic_demo.pt`. |
| Out of memory on the free tier | Lower the image size you accept, or move to a larger instance. The model itself is tiny; memory use comes from PyTorch and large uploads. |
| CI Docker job fails but local run works | Read the failing step. The Docker build was written but could not be executed in the environment where this guide was prepared, so the first GitHub run is its real test. |
| Push to Hugging Face asks for a password repeatedly | Use the access token as the password, with the Write role. |
| Demo output looks wrong on a real scan | Expected: the bundled model was trained on synthetic images only. |

---

## 7. Swap in a real-data model (after Milestone 8)

1. Train on real data:

```powershell
python train.py --data folder --data_dir dataset --n_segments 100 --iters 1000 --out runs/real
```

2. Copy the best checkpoint into the repo with a clear name and commit it:

```powershell
copy runs\real\best.pt checkpoints\real_model.pt
git add -f checkpoints/real_model.pt
```

3. Point the app at it:
   - Locally: `set MODEL_PATH=checkpoints/real_model.pt` (PowerShell: `$env:MODEL_PATH="checkpoints/real_model.pt"`), then `python app.py`.
   - On Hugging Face: Space → **Settings → Variables and secrets → New variable** `MODEL_PATH` = `checkpoints/real_model.pt`.
   - On Render: **Environment → Add variable** `MODEL_PATH`.
4. Update the text in `app.py` (the description box) so it no longer says "synthetic only", and update the README results and status.

**Privacy:** a public demo must not receive patient-identifiable images. Say clearly on the page that uploads are for research and testing, use de-identified public data for examples, and keep the "not for medical use" notice.

---

## 8. Optional: auto-deploy to Hugging Face from GitHub

After the manual deploy works, you can push to the Space automatically on each commit to `main`. Add a repository secret `HF_TOKEN` (the Write token) in GitHub → Settings → Secrets and variables → Actions, then add a step like this to a new workflow. This snippet is a starting point and has not been run for you:

```yaml
name: Deploy to Hugging Face
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Push to Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          cp deploy/huggingface/SPACE_README.md README.md
          git config user.email "ci@example.com"
          git config user.name "ci"
          git add -A && git commit -m "Space README" || true
          git push --force https://<your-username>:$HF_TOKEN@huggingface.co/spaces/<your-username>/graph-rl-segmentation main
```

The force push is to the **Space** repository only (it overwrites the Space's README with the Space version), never to your GitHub repository.

---

## 9. Deployment checklist

- [ ] `python -m pytest -q` passes locally
- [ ] `python app.py` works and example images give sensible overlays
- [ ] Code pushed to GitHub and the **CI** workflow is green
- [ ] Space or Render service shows **Running**
- [ ] The public URL opens in a private browser window and an example image works
- [ ] The live URL is added to the README ("Live demo")
- [ ] The page states that the model is a research prototype and not for medical use
