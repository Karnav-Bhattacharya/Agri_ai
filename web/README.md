# Agri AI Chatbot Web UI

## How to Run

1. **Install dependencies** (in the web folder):

   ```sh
   pip install flask
   ```

   Also ensure all dependencies for `main.py` are installed in the parent folder.

2. **Set your environment variables** (in the parent folder):

   - `GROQ_API_KEY` (required)
   - Optionally: `GROQ_MODEL`, `GROQ_TEMPERATURE`, `GROQ_MAX_TOKENS`

3. **Start the server**:

   ```sh
   cd web
   python server.py
   ```

   The app will be available at [http://localhost:7860](http://localhost:7860)

4. **Chat!**

## Notes

- The backend uses the same logic as the CLI agent in `main.py`.
- The web UI is green-themed and mobile-friendly.
- If you add new datasets, restart the server to pick them up.
