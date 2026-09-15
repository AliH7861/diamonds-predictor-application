import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

# Import things from the prediction.py folder
from .prediction import DEFAULT_MODEL_DIR, load_best_model, predict_prices


# Create a Local-Only HTTP API Around the Saved Regression Pipeline
def create_server(model_dir=DEFAULT_MODEL_DIR, port=8766):
    # Load the best model from the specified directory and prepare it for predictions.
    run = load_best_model(model_dir)

    # Define a Custom HTTP Request Handler for the Price Prediction API
    class PriceHandler(BaseHTTPRequestHandler):

        # Helper Method to Send JSON Responses
        def respond(self, status, body):
            content = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        # Handle GET Requests for Health Check and Model Info
        def do_GET(self):

            if self.path == "/health":
                self.respond(200, {"status": "ready", "target": "price"})

            elif self.path == "/model":
                self.respond(200, {
                    "algorithm": run["algorithm"],
                    "feature_set": run["feature_set"],
                    "required_inputs": ["carat", "cut", "color", "clarity", "depth", "table", "x",
                     "y", "z"],
                })

            else:
                self.respond(404, {"error": "Use GET /health, GET /model, or POST /predict."})

        # Handle POST Requests for Price Predictions
        def do_POST(self):
            # Check if the Request Path is /predict, Otherwise Return a 404 Error
            if self.path != "/predict":
                self.respond(404, {"error": "Use POST /predict."})
                return

            try:
                # Read the Request Body and Parse it as JSON, Ensuring It is Under 1 MB
                size = int(self.headers.get("Content-Length", "0"))

                if size <= 0 or size > 1_000_000:
                    raise ValueError("Request JSON must be present and under 1 MB.")

                records = json.loads(self.rfile.read(size))
                self.respond(200, {"predictions": predict_prices(run, records)})

            # Handle Any Errors That Occur During Prediction and Return Appropriate Responses
            except (ValueError, TypeError, KeyError) as error:
                self.respond(400, {"error": str(error)})

            # Handle Any Other Unexpected Errors and Return a 500 Internal Server Error
            except Exception:
                self.respond(500, {"error": "Prediction failed in the saved model environment."})

        # Override the log_message Method to Suppress Logging to the Console
        def log_message(self, format, *args):
            pass
    
    # Return the HTTP Server Instance Bound to the Specified Port
    return HTTPServer(("127.0.0.1", port), PriceHandler)


# Start Without Blocking a Notebook Cell
def start_prediction_api(model_dir=DEFAULT_MODEL_DIR, port=8766):
    server = create_server(model_dir, port)

    # Start the Server in a Separate Thread to Avoid Blocking the Main Thread
    Thread(target=server.serve_forever, daemon=True).start()

    print(f"Regression API ready at http://127.0.0.1:{server.server_port}")

    return server


if __name__ == "__main__":
    # Parse Command-Line Arguments for Model Directory and Port
    parser = argparse.ArgumentParser(description="Serve the saved diamond price model locally.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--port", type=int, default=8766)
    
    # Create and Start the Regression API Server
    args = parser.parse_args()
    server = create_server(args.model_dir, args.port)
    print(f"Regression API ready at http://127.0.0.1:{server.server_port}")

    # Keep the Server Running Until Interrupted
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
