import os
from flask import Flask, request, jsonify, send_from_directory
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import main as agri_main

app = Flask(__name__, static_folder='.', static_url_path='')

# Load and cache LLM, router, tools, etc. on startup
llm = None
router_chain = None
all_tools = None
vectorstores = None

def init_agent():
    global llm, router_chain, all_tools, vectorstores
    if llm is not None:
        return
    # Find dataset folders
    datasets_dir = os.path.join(os.path.dirname(__file__), '../datasets')
    dataset_folders = {}
    if os.path.exists(datasets_dir):
        for sub in os.listdir(datasets_dir):
            sub_path = os.path.join(datasets_dir, sub)
            if os.path.isdir(sub_path):
                dataset_folders[sub] = sub_path
    vectorstores = {name: agri_main.build_vectorstore_for_dataset(name, folder, persist_root="../faiss_dbs") for name, folder in dataset_folders.items()}
    llm = agri_main.init_groq_llm()
    router_chain = agri_main.make_router_chain(llm)
    all_tools = agri_main.make_lc_tools_for_datasets(vectorstores)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('.', path)

@app.route('/api/chat', methods=['POST'])
def chat():
    init_agent()
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    dataset_names = list(vectorstores.keys())
    out = agri_main.handle_query_router(router_chain, all_tools, llm, query, dataset_names)
    if out.get('clarify'):
        return jsonify({'answer': out.get('message', 'Please clarify your question.')})
    return jsonify({'answer': out.get('answer', '[No answer]')})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=True)
