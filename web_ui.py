import cgi
import html
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from main import run_pipeline

BASE_DIR = Path(__file__).resolve().parent
WEB_RUNS_DIR = BASE_DIR / "web_runs"
UPLOADS_DIR = WEB_RUNS_DIR / "uploads"
RESULTS_DIR = WEB_RUNS_DIR / "results"

for p in [WEB_RUNS_DIR, UPLOADS_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def slugify_filename(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    return safe or "upload.jpg"


def render_home(message: str = "") -> bytes:
    msg_html = f"<p class='msg'>{html.escape(message)}</p>" if message else ""
    page = """<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1.0' />
  <title>Galaxy DIP Demo</title>
  <style>
    :root {
      --bg1: #0f2027;
      --bg2: #203a43;
      --bg3: #2c5364;
      --card: #ffffff;
      --text: #162027;
      --accent: #e76f51;
      --accent2: #264653;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: 'Trebuchet MS', 'Segoe UI', sans-serif;
      color: var(--text);
      background: linear-gradient(140deg, var(--bg1), var(--bg2), var(--bg3));
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 22px;
    }
    .card {
      width: min(900px, 96vw);
      background: var(--card);
      border-radius: 18px;
      padding: 24px;
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.28);
    }
    h1 { margin: 0 0 8px; color: var(--accent2); }
    p { margin: 0 0 14px; }
    .msg {
      background: #fff5d9;
      border: 1px solid #f2cf66;
      padding: 10px;
      border-radius: 10px;
      margin-bottom: 14px;
    }
    .dropzone {
      border: 2px dashed #88a3b7;
      border-radius: 14px;
      padding: 32px;
      text-align: center;
      margin: 14px 0;
      transition: 0.2s ease;
      background: #f7fafc;
    }
    .dropzone.drag {
      border-color: var(--accent);
      background: #fff0eb;
    }
    .controls {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      margin: 14px 0;
    }
    label { display: block; font-weight: 600; margin-bottom: 6px; }
    input[type='number'], input[type='file'] { width: 100%; padding: 8px; }
    button {
      border: 0;
      background: var(--accent);
      color: white;
      padding: 10px 16px;
      border-radius: 10px;
      font-weight: 700;
      cursor: pointer;
    }
    .hint { font-size: 0.92rem; color: #3a4651; }
  </style>
</head>
<body>
  <div class='card'>
    <h1>Galaxy DIP Pipeline</h1>
    <p>Upload or drag-drop an image to run enhancement, segmentation, ML clustering, and matplotlib reports.</p>
    __MSG__
    <form action='/process' method='post' enctype='multipart/form-data'>
      <div id='dropzone' class='dropzone'>
        <strong>Drop image here</strong><br/>
        or choose below
      </div>
      <input id='image-input' type='file' name='image' accept='.jpg,.jpeg,.png,.bmp,.tif,.tiff' required />
      <div class='controls'>
        <div>
          <label for='max_size'>Max side for processing</label>
          <input id='max_size' type='number' name='max_size' value='1600' min='512' max='5000' />
        </div>
        <div>
          <label for='clusters'>K-means clusters</label>
          <input id='clusters' type='number' name='clusters' value='5' min='2' max='12' />
        </div>
      </div>
      <button type='submit'>Run Pipeline</button>
      <p class='hint'>All outputs are saved in <code>web_runs/results/...</code>.</p>
    </form>
  </div>

  <script>
    const dz = document.getElementById('dropzone');
    const input = document.getElementById('image-input');

    ['dragenter', 'dragover'].forEach(ev => {
      dz.addEventListener(ev, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dz.classList.add('drag');
      });
    });

    ['dragleave', 'drop'].forEach(ev => {
      dz.addEventListener(ev, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dz.classList.remove('drag');
      });
    });

    dz.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        input.files = files;
      }
    });
  </script>
</body>
</html>"""
    return page.replace("__MSG__", msg_html).encode("utf-8")


def render_results(run_id: str, outputs: dict[str, Path]) -> bytes:
    rows = []
    for key, path in outputs.items():
        rel = path.relative_to(WEB_RUNS_DIR)
        label = html.escape(key.replace("_", " ").title())
        rows.append(
            """
            <div class='item'>
              <h3>{label}</h3>
              <img src='/artifact/{src}' alt='{label}' />
              <p><code>{path}</code></p>
            </div>
            """.format(
                label=label,
                src=rel.as_posix(),
                path=html.escape(str(path)),
            )
        )

    page = """<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1.0' />
  <title>Results - __RUNID__</title>
  <style>
    body {
      margin: 0;
      font-family: 'Trebuchet MS', 'Segoe UI', sans-serif;
      background: #f2f5f7;
      color: #1b2630;
      padding: 18px;
    }
    .top { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
    .btn {
      text-decoration: none;
      background: #264653;
      color: white;
      padding: 8px 12px;
      border-radius: 8px;
      font-weight: 700;
    }
    .grid {
      margin-top: 14px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 14px;
    }
    .item {
      background: white;
      padding: 12px;
      border-radius: 12px;
      box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    }
    .item img {
      width: 100%;
      border-radius: 8px;
      border: 1px solid #d9e0e6;
    }
    h2, h3 { margin-top: 0; }
    p { margin-bottom: 0; font-size: 0.9rem; }
    code { word-break: break-all; }
  </style>
</head>
<body>
  <div class='top'>
    <a href='/' class='btn'>Upload Another</a>
    <h2>Run ID: __RUNID__</h2>
  </div>
  <div class='grid'>
    __ROWS__
  </div>
</body>
</html>"""
    page = page.replace("__RUNID__", html.escape(run_id))
    page = page.replace("__ROWS__", "".join(rows))
    return page.encode("utf-8")


class DIPHandler(BaseHTTPRequestHandler):
    def _send_html(self, content: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/":
            self._send_html(render_home())
            return

        if self.path.startswith("/artifact/"):
            rel = self.path.removeprefix("/artifact/")
            file_path = (WEB_RUNS_DIR / rel).resolve()
            if WEB_RUNS_DIR.resolve() not in file_path.parents and file_path != WEB_RUNS_DIR.resolve():
                self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
                return
            if not file_path.exists() or not file_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return

            ext = file_path.suffix.lower()
            content_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".bmp": "image/bmp",
                ".webp": "image/webp",
            }.get(ext, "application/octet-stream")
            self._send_bytes(file_path.read_bytes(), content_type)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:
        if self.path != "/process":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            },
        )

        if "image" not in form or not getattr(form["image"], "file", None):
            self._send_html(render_home("Please upload an image file."), status=400)
            return

        image_item = form["image"]
        filename = slugify_filename(getattr(image_item, "filename", "upload.jpg"))

        try:
            max_size = int(form.getfirst("max_size", "1600"))
            clusters = int(form.getfirst("clusters", "5"))
            max_size = min(max(max_size, 512), 5000)
            clusters = min(max(clusters, 2), 12)
        except ValueError:
            self._send_html(render_home("Invalid numeric values for max_size/clusters."), status=400)
            return

        run_id = time.strftime("%Y%m%d_%H%M%S")
        upload_dir = UPLOADS_DIR / run_id
        result_dir = RESULTS_DIR / run_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        upload_path = upload_dir / filename
        raw = image_item.file.read()
        upload_path.write_bytes(raw)

        try:
            outputs = run_pipeline(upload_path, result_dir, max_size=max_size, clusters=clusters)
        except Exception as exc:
            self._send_html(render_home(f"Pipeline error: {exc}"), status=500)
            return

        self._send_html(render_results(run_id, outputs))


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), DIPHandler)
    print("Web UI running at http://127.0.0.1:8000")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
