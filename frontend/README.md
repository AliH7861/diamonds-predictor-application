# React frontend

This Vite/React interface connects to the existing local Python assistant API. It provides the visual shell, conversation history, compact conversation state, streaming responses, and natural-language match cards. Prediction and knowledge-center screens can be expanded later without changing the backend contract.

## Run locally

Terminal 1, from the repository root:

```powershell
python -m src.assistant.api --port 8770
```

Terminal 2:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

Vite proxies `/assistant-api` to `http://127.0.0.1:8770`. For a separately hosted frontend, copy `.env.example` to `.env` and set `VITE_ASSISTANT_API_URL` to the public HTTPS backend URL. Set the same optional token in `VITE_ASSISTANT_API_TOKEN` and `DIAMOND_ASSISTANT_API_TOKEN`.

## Production build

```powershell
npm run build
npm run preview
```
