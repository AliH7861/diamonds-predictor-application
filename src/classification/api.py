import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from .prediction import DEFAULT_MODEL_DIR, load_best_model, predict_diamonds

# Create a Local HTTP API for the Saved Classification Model
def create_server(model_dir=DEFAULT_MODEL_DIR, port=8765):
    bundle = load_best_model(model_dir)

    class PredictionHandler(BaseHTTPRequestHandler):
        def respond(self, status, data):
            content = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path == "/model":
                self.respond(200, bundle["metadata"])
            elif self.path == "/health":
                self.respond(200, {"status": "ready", "model": bundle["metadata"]["model_name"]})
            else:
                self.respond(404, {"error": "Use GET /model, GET /health, or POST /predict."})

        def do_POST(self):
            if self.path != "/predict":
                self.respond(404, {"error": "Use POST /predict."})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 1_000_000:
                    raise ValueError("Request body must contain JSON and be under 1 MB.")
                records = json.loads(self.rfile.read(size))
                predictions = predict_diamonds(bundle, records)
                self.respond(200, {"model": bundle["metadata"]["model_name"], "predictions": predictions})
            except (ValueError, TypeError, KeyError) as error:
                self.respond(400, {"error": str(error)})
            except Exception:
                self.respond(500, {"error": "Prediction failed. Check the saved model and its Python environment."})

        def log_message(self, format, *args):
            pass

    return HTTPServer(("127.0.0.1", port), PredictionHandler)


# Start the API Without Blocking the Notebook
def start_prediction_api(model_dir=DEFAULT_MODEL_DIR, port=8765):
    server = create_server(model_dir, port)
    Thread(target=server.serve_forever, daemon=True).start()
    print(f"Classification API ready at http://127.0.0.1:{server.server_port}")
    return server


# Start the Same API From a Terminal
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the best saved diamond clarity classifier locally.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = create_server(args.model_dir, args.port)
    print(f"Classification API ready at http://127.0.0.1:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
