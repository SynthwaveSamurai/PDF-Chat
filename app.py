#3.9.18

import os
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, send_from_directory
import PyPDF2
from transformers import AutoTokenizer, AutoModel
import torch
import nltk
import sqlite3
import numpy as np
from numpy.linalg import norm
import json
import openai
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from urllib.parse import unquote

openai.api_key = os.getenv("OPENAI_API_KEY")

nltk.download('punkt')
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
UPLOAD_FOLDER = 'uploads' 

def embed_question(question):
    #print(f"Embedding question: {question}")
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    question_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    
    #print("Frage-Embedding:", question_embedding)
    return question_embedding

def find_all_pdfs(base_directory):
    pdf_files = []
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith('.pdf'):
                relative_path = os.path.relpath(root, base_directory)
                pdf_files.append(os.path.join(relative_path, file))
    return pdf_files

def get_most_similar_embeddings_for_course(question, course_id, min_similarity_threshold=0.8):
    #print(f"Fetching most similar embeddings for course with ID {course_id}")
    question_embedding = np.array(embed_question(question)).reshape(1, -1)  # Konvertiere die Frageembedding in eine 2D-Form

    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    try: 
        c.execute("SELECT document, text_chunk, embedding FROM embeddings WHERE course_id = ?", (course_id,))
        rows = c.fetchall()
        #print(f"Fetched {len(rows)} rows from database for course_id {course_id}.")
    finally:
        conn.close()

    similarities = []
    for idx, (doc, text, embedding_str) in enumerate(rows):
        try:
            # Lade das Dokumentembedding und konvertiere es in eine 2D-Form
            embedding = np.array(json.loads(embedding_str)).reshape(1, -1)
            # Berechne die Kosinusähnlichkeit
            similarity = cosine_similarity(question_embedding, embedding)[0][0]
            similarities.append((similarity, doc, text))
            #print(f"Processed row {idx}: doc={doc}, similarity={similarity}")
        except Exception as e:
            print(f"Error decoding JSON for document {doc}: {e}")
            continue

    # Sortiere die Ergebnisse basierend auf der Ähnlichkeit
    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities [:5]

@app.route('/api/highlight_text', methods=['POST'])
def highlight_text():
    data = request.get_json()
    course_id = data.get('course_id')
    question = data.get('question')

    if not question or not course_id:
        return jsonify({"error": "Invalid data"}), 400

    most_similar = get_most_similar_embeddings_for_course(question, course_id)

    pdf_text_pairs = []
    for _, document, text_chunk in most_similar:
        pdf_text_pairs.append({"document": document, "text_chunk": text_chunk})

    return jsonify(pdf_text_pairs)

def get_schools_by_university(university_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT id, name FROM school WHERE university_id = ?", (university_id,))
    schools = c.fetchall()
    conn.close()
    return schools

def get_chairs_by_school(school_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT id, name FROM chair WHERE school_id = ?", (school_id,))
    chairs = c.fetchall()
    conn.close()
    return chairs

def get_courses_by_chair(chair_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT id, name FROM course WHERE chair_id = ?", (chair_id,))
    courses = c.fetchall()
    conn.close()
    return courses

# Erstelle die Datenbankverbindung und Tabelle
def init_db():
    try:
        # Verbindung zur Datenbank herstellen
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        print("Datenbankverbindung hergestellt.")  # Debug-Ausgabe

        # Tabellen erstellen
        c.execute('''CREATE TABLE IF NOT EXISTS university (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        )''')
        print("Tabelle 'university' erstellt oder existiert bereits.")  # Debug-Ausgabe

        c.execute('''CREATE TABLE IF NOT EXISTS school (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            university_id INTEGER,
            FOREIGN KEY(university_id) REFERENCES university(id)
        )''')
        print("Tabelle 'school' erstellt oder existiert bereits.")  # Debug-Ausgabe

        c.execute('''CREATE TABLE IF NOT EXISTS chair (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            school_id INTEGER,
            FOREIGN KEY(school_id) REFERENCES school(id)
        )''')
        print("Tabelle 'chair' erstellt oder existiert bereits.")  # Debug-Ausgabe

        c.execute('''CREATE TABLE IF NOT EXISTS course (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            chair_id INTEGER,
            FOREIGN KEY(chair_id) REFERENCES chair(id)
        )''')
        print("Tabelle 'course' erstellt oder existiert bereits.")  # Debug-Ausgabe

        c.execute('''CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY,
            document TEXT NOT NULL,
            text_chunk TEXT NOT NULL,
            embedding TEXT NOT NULL,
            university_id INTEGER,
            school_id INTEGER,
            chair_id INTEGER,
            course_id INTEGER,
            FOREIGN KEY(university_id) REFERENCES university(id),
            FOREIGN KEY(school_id) REFERENCES school(id),
            FOREIGN KEY(chair_id) REFERENCES chair(id),
            FOREIGN KEY(course_id) REFERENCES course(id)
        )''')
        print("Tabelle 'embeddings' erstellt oder existiert bereits.")  # Debug-Ausgabe

        # Änderungen übernehmen
        conn.commit()
        print("Alle Tabellen erfolgreich erstellt oder existierten bereits.")  # Debug-Ausgabe

    except sqlite3.Error as e:
        print("Datenbankinitialisierungsfehler: ", e.args[0])  # Debug-Ausgabe über den Fehler

    finally:
        # Verbindung zur Datenbank schließen
        if conn:
            conn.close()
            print("Datenbankverbindung geschlossen.")  # Debug-Ausgabe

def get_all_universities():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT id, name FROM university")  # Annahme: Die Tabelle 'university' hat Spalten 'id' und 'name'
    universities = c.fetchall()
    conn.close()
    return universities

# Füge Text, Datei und Embedding in die Tabelle ein
def insert_into_db(document, text_chunk, embedding, university_id=None, school_id=None, chair_id=None, course_id=None):
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        # Speichere das Embedding als JSON-freundlicher String
        embedding_str = json.dumps(embedding.tolist())
        c.execute("""
            INSERT INTO embeddings (document, text_chunk, embedding, university_id, school_id, chair_id, course_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (document, text_chunk, embedding_str, university_id, school_id, chair_id, course_id))
        conn.commit()
    except Exception as e:
        print(f"Error inserting into DB: {e}")
    finally:
        conn.close()

#Text aus PDF extrahieren
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        text = ""
        for page in reader.pages:
            text += page.extract_text() if page.extract_text() else ""
    
    return text

def split_text_into_hierarchical_chunks(text, chunk_size=50, overlap=25):
    # Worttokenisierung
    words = nltk.word_tokenize(text)
    chunks = []

    # Stelle sicher, dass die Werte für Chunk-Größe und Überlappung gültig sind
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Ungültige Werte für chunk_size oder overlap angegeben.")

    step = chunk_size - overlap  # Schrittweite zur Erstellung der überlappenden Chunks

    for start in range(0, len(words), step):
        end = start + chunk_size
        # Füge den Textchunk zur Liste der Chunks hinzu
        chunk = words[start:end]
        if chunk:
            chunks.append(' '.join(chunk))

    return chunks

#Text in Embeddings umwandeln
def generate_embeddings(text_chunks):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    
    embeddings = []
    for index, chunk in enumerate(text_chunks): 
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append((index, embedding))
    
    return embeddings

chaten_history = []

def get_text_chunks_for_course(course_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT text_chunk FROM embeddings WHERE course_id = ?", (course_id,))
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]

def get_text_chunks_and_docs(course_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT text_chunk, document FROM embeddings WHERE course_id = ?", (course_id,))
    rows = c.fetchall()
    conn.close()
    print("Fetched text chunks and docs:")
    for row in rows:
        print(row)

    text_chunks_with_docs = [(row[0], row[1]) for row in rows]
    return text_chunks_with_docs

@app.route('/api/universities', methods=['GET', 'POST'])
def api_universities():
    if request.method == 'GET':
        print("GET-Anfrage für Universitäten empfangen.")
        universities = get_all_universities()
        print("Aktuelle Universitäten in der DB:", universities)
        return jsonify(universities)

    elif request.method == 'POST':
        data = request.get_json()
        print("POST-Daten empfangen:", data)
        
        if not data:
            return jsonify({"error": "Keine gültigen Daten"}), 400

        name = data.get('name')
        if name:
            try:
                conn = sqlite3.connect('database.db')
                c = conn.cursor()
                c.execute("INSERT INTO university (name) VALUES (?)", (name,))
                conn.commit()
                conn.close()
                print(f"Universität '{name}' erfolgreich hinzugefügt.")
                return jsonify(success=True)
            except sqlite3.Error as e:
                print(f"Fehler beim Hinzufügen der Universität: {e.args[0]}")
                return jsonify(success=False), 500
        else:
            return jsonify({"error": "Ungültige Daten für die Universität"}), 400

@app.route('/api/schools', methods=['POST'])
def add_school():
    data = request.get_json()
    print("POST-Daten empfangen für die Schule:", data)
    
    if not data:
        return jsonify({"error": "Keine gültigen Daten"}), 400
    
    name = data.get('name')
    university_id = data.get('university_id')

    if name and university_id:
        try:
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute("INSERT INTO school (name, university_id) VALUES (?, ?)", (name, university_id))
            conn.commit()
            conn.close()
            print(f"Schule '{name}' erfolgreich hinzugefügt.")
            return jsonify(success=True)
        except sqlite3.Error as e:
            print(f"Fehler beim Hinzufügen der Schule: {e.args[0]}")
            return jsonify(success=False), 500
    else:
        return jsonify({"error": "Ungültige Daten für die Schule"}), 400

@app.route('/api/chairs', methods=['POST'])
def add_chair():
    data = request.get_json()
    print("POST-Daten empfangen für den Lehrstuhl:", data)
    
    if not data:
        return jsonify({"error": "Keine gültigen Daten"}), 400

    name = data.get('name')
    school_id = data.get('school_id')

    if name and school_id:
        try:
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute("INSERT INTO chair (name, school_id) VALUES (?, ?)", (name, school_id))
            conn.commit()
            conn.close()
            print(f"Lehrstuhl '{name}' erfolgreich hinzugefügt.")
            return jsonify(success=True)
        except sqlite3.Error as e:
            print(f"Fehler beim Hinzufügen des Lehrstuhls: {e.args[0]}")
            return jsonify(success=False), 500
    else:
        return jsonify({"error": "Ungültige Daten für den Lehrstuhl"}), 400

@app.route('/api/courses', methods=['POST'])
def add_course():
    data = request.get_json()
    print("POST-Daten empfangen für den Kurs:", data)
    
    if not data:
        return jsonify({"error": "Keine gültigen Daten"}), 400

    name = data.get('name')
    chair_id = data.get('chair_id')

    if name and chair_id:
        try:
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute("INSERT INTO course (name, chair_id) VALUES (?, ?)", (name, chair_id))
            conn.commit()
            conn.close()
            print(f"Kurs '{name}' erfolgreich hinzugefügt.")
            return jsonify(success=True)
        except sqlite3.Error as e:
            print(f"Fehler beim Hinzufügen des Kurses: {e.args[0]}")
            return jsonify(success=False), 500
    else:
        return jsonify({"error": "Ungültige Daten für den Kurs"}), 400

@app.route('/api/schools/<int:university_id>')
def api_schools(university_id):
    schools = get_schools_by_university(university_id)
    return jsonify(schools)

@app.route('/api/chairs/<int:school_id>')
def api_chairs(school_id):
    chairs = get_chairs_by_school(school_id)
    return jsonify(chairs)

@app.route('/api/courses/<int:chair_id>')
def api_courses(chair_id):
    courses = get_courses_by_chair(chair_id)
    return jsonify(courses)

@app.route('/chaten', methods=['GET', 'POST'])
def chaten():
    global chaten_history
    if request.method == 'POST':
        question = request.form['question']
        most_similar = get_most_similar_embeddings_for_course(question)
        chaten_history.append((question, most_similar))
    
    return render_template('chaten.html', chaten_history=chaten_history)

@app.route('/verwalten', methods=['GET', 'POST'])
def verwalten():
    if request.method == 'POST':
        if 'question' in request.form:
            question = request.form['question']
            most_similar = get_most_similar_embeddings_for_course(question)
            return render_template('verwalten.html', results=most_similar)
        else:
            return "Fehler: Kein Fragefeld ausgefüllt.", 400

    return render_template('verwalten.html', results=None)

@app.route('/universities', methods=['GET'])
def universities():
    universities = get_all_universities()
    return render_template('universities.html', universities=universities)

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')
    course_id = data.get('course_id')

    if not question or not course_id:
        return jsonify({"error": "Invalid data"}), 400

    most_similar = get_most_similar_embeddings_for_course(question, course_id)
    text_chunks_with_docs = get_text_chunks_and_docs(course_id)
    
    # Füge hier den course_id beim Aufruf hinzu
    answer, used_chunk, used_doc, highlight_snippet= generate_answer_with_openai(question, most_similar, text_chunks_with_docs, course_id)

    if not used_doc:
        return jsonify({
            "error": "Keine passende Dokumentation gefunden."
        })

    return jsonify({
        "answer": answer,
        "similar_texts": [text for _, _, text in most_similar],
        "documents": list(set([doc for _, doc, _ in most_similar])),
        "used_chunk": used_chunk,
        "used_doc": used_doc,
        "highlight_snippet": highlight_snippet
    })

def check_database_contents():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT document, text_chunk, embedding FROM embeddings")
    rows = c.fetchall()
    conn.close()
    if not rows:
        print("Datenbank ist leer oder keine Embeddings gefunden.")
    else:
        for doc, text, embedding_str in rows:
            try:
                embedding = np.array(json.loads(embedding_str))
                print(f"Dokument: {doc}, Textausschnitt: {text[:75]}..., Embedding-Länge: {len(embedding)}")
            except Exception as e:
                print(f"Fehler beim Laden von Embeddings für Dokument {doc}: {e}")

def prepare_search_models(text_chunks_with_docs):
    # Extrahiere nur die Text-Chunks, nicht die zugehörigen Dokumentinfos
    text_chunks = [chunk for chunk, _ in text_chunks_with_docs]
    
    # TF-IDF Vektorisierer vorbereiten
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_chunks)

    # BM25 vorbereiten
    tokenized_chunks = [nltk.word_tokenize(chunk.lower()) for chunk in text_chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    return tfidf_vectorizer, tfidf_matrix, bm25, tokenized_chunks

def extract_used_chunk_via_bm25(answer, text_chunks_with_docs):
    # Tokenisiere alle Chunks für BM25
    tokenized_chunks = [nltk.word_tokenize(chunk.lower()) for chunk, _ in text_chunks_with_docs]
    bm25 = BM25Okapi(tokenized_chunks)

    # Tokenisiere und berechne die Ähnlichkeit der Antwort mit den Chunks
    tokenized_answer = nltk.word_tokenize(answer.lower())
    scores = bm25.get_scores(tokenized_answer)

    # Finde den Index des Chunks mit dem höchsten Score
    best_chunk_idx = scores.argmax()
    best_chunk, best_doc = text_chunks_with_docs[best_chunk_idx]
    print(f"Bester gematchter chunk: '{best_chunk}' in document: '{best_doc}' mit score: {scores[best_chunk_idx]}")

    return best_chunk, best_doc

def get_top_indices(scores, top_n=5):
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]

def search_with_bm25(bm25, tokenized_chunks, query, top_n=5):
    tokenized_query = nltk.word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    return get_top_indices(scores, top_n)

def search_with_tfidf(tfidf_vectorizer, tfidf_matrix, query, top_n=5):
    query_vect = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vect, tfidf_matrix).flatten()
    return get_top_indices(cosine_similarities, top_n)

def generate_combined_context(most_similar, text_chunks_with_docs, question):
    tfidf_vectorizer, tfidf_matrix, bm25, tokenized_chunks = prepare_search_models(text_chunks_with_docs)

    tfidf_indices = search_with_tfidf(tfidf_vectorizer, tfidf_matrix, question)
    bm25_indices = search_with_bm25(bm25, tokenized_chunks, question)

    # Nur die chunks aus den Textschlüssen extrahieren
    tfidf_context = [text_chunks_with_docs[i][0] for i in tfidf_indices]  # Nur chunks extrahieren
    bm25_context = [text_chunks_with_docs[i][0] for i in bm25_indices]    # Nur chunks extrahieren
    embedding_context = [text for _, _, text in most_similar]  # Annahme, dass dies die erwarteten Strings sind

    # Kombiniert die Kontexte
    context = "\n".join(set(tfidf_context + bm25_context + embedding_context))
    return context

def generate_answer_with_openai(question, most_similar, text_chunks_with_docs, course_id):
    context = generate_combined_context(most_similar, text_chunks_with_docs, question)

    if not context:
        print("Kein Kontext für die Frage vorhanden.")
        return "Kein relevanter Kontext gefunden", None, None, None

    prompt = f"Antwort auf folgende Frage basierend auf den Informationen: {context}\n\nFrage: {question}\nAntwort:"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=750
        )

        message_content = response['choices'][0]['message']['content'].strip()

        # Verwende BM25, um den am besten passenden Chunk zu extrahieren
        used_chunk, used_doc = extract_used_chunk_via_bm25(message_content, text_chunks_with_docs)

        highlight_snippet = extract_highlight_snippet(question, message_content, used_chunk)

        # Hole die IDs aus der Datenbank basierend auf dem Dokumentnamen und der course_id
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("""
            SELECT university_id, school_id, chair_id
            FROM embeddings
            WHERE course_id = ? AND document = ?
            LIMIT 1
        """, (course_id, used_doc))
        result = c.fetchone()
        conn.close()

        if result:
            university_id, school_id, chair_id = result
            # Erstelle den Pfad
            doc_path = f"uploads/{university_id}\\{school_id}\\{chair_id}\\{course_id}\\{used_doc}"
            print(f"Verwendeter Text-Chunk: '{used_chunk}' aus dem Dokument '{doc_path}'")
            return message_content, used_chunk, doc_path, highlight_snippet
        
        print("Kein passender Eintrag gefunden.")

        return message_content, used_chunk, None, highlight_snippet

    except Exception as e:
        print(f"OpenAI API Error: {str(e)}")
        return "Es gab ein Problem bei der Verarbeitung der Anfrage.", None, None, None
    
def extract_highlight_snippet(question: str, message_content: str, used_chunk: str) -> str:
    prompt = f"""Aus dem verwendeten Textausschnitt sollst Du **nur den sehr kurzen Satzteil** zurückgeben, der die Antwort auf die Frage beinhaltet. Also konkrete Inforamtionen und keine Verweise. Also die Schlüsselinformationen. Frage: {question}; Antwort: {message_content}; Verwendeter Textausschnitt: {used_chunk}. Versuche das dieser Teil möglichst kurz ist. Verwende im Idealfall für 1-5 Wörter. Du darfst nur Teile verwenden die exakt so im Verwendeter Textausschnitt vorkommen, da ich diese Wörter dann später über die Suchfunktion finden will. Du darfst demnach auch nicht die Reihenfolge ändern oder andere Korrekturen vornehmen."""
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role":"system", "content":"Du extrahierst kurze Highlight-Snippets. Also den kürzesten Abschnitt mit allen wichtigen Informationen."},
            {"role":"user",   "content":prompt}
        ],
        max_tokens=50,
        temperature=0.0
    )

    highlight_snippet = resp.choices[0].message.content.strip().strip('"\'')
    print(f"📝 Highlight-Snippet: {highlight_snippet}")
    return highlight_snippet

def execute_query(query, params=()):
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return rows

@app.route('/api/documents', methods=['POST'])
def get_documents():
    data = request.get_json()

    conditions = []
    params = []

    if data.get('university_id'):
        conditions.append("university_id = ?")
        params.append(data['university_id'])
    if data.get('school_id'):
        conditions.append("school_id = ?")
        params.append(data['school_id'])
    if data.get('chair_id'):
        conditions.append("chair_id = ?")
        params.append(data['chair_id'])
    if data.get('course_id'):
        conditions.append("course_id = ?")
        params.append(data['course_id'])

    query = "SELECT DISTINCT university_id, school_id, chair_id, course_id, document FROM embeddings"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    results = execute_query(query, tuple(params))

    document_paths = [
        os.path.join(str(row['university_id']), str(row['school_id']), str(row['chair_id']), str(row['course_id']), row['document'])
        for row in results
    ]

    return jsonify(document_paths)

@app.route('/api/documents/<int:university_id>/<int:school_id>/<int:chair_id>/<int:course_id>', methods=['GET'])
def list_documents(university_id, school_id, chair_id, course_id):
    # Erstellen Sie den Pfad basierend auf den IDs
    directory = os.path.join(UPLOAD_FOLDER, str(university_id), str(school_id), str(chair_id), str(course_id))
    
    if not os.path.exists(directory):
        return jsonify({"error": "Dokumentverzeichnis nicht gefunden"}), 404

    # Listet alle PDFs im Verzeichnis auf
    documents = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    return jsonify(documents)

@app.route('/uploads/<path:filepath>', methods=['GET'])
def download_file(filepath):
    decoded_filepath = unquote(filepath)
    directory_path, filename = os.path.split(decoded_filepath)
    full_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        directory_path.replace('/', os.path.sep),
        filename
    )

    # Logging und Existenzüberprüfung
    app.logger.info(f"Trying to access: {full_path}")
    if not os.path.exists(full_path):
        app.logger.error(f"File not found: {full_path}")
        return jsonify({"error": "File not found"}), 404

    # 4. PDF inline ausliefern, ohne Caching
    try:
        return send_file(
            full_path,           # hier den absoluten Pfad verwenden
            mimetype='application/pdf',
            as_attachment=False, # inline im Browser darstellen
            conditional=False,     # unterstützt Range-Requests
            max_age=0            # Cache-Control: no-cache
        )
    except Exception as e:
        app.logger.error(f"Error sending file {full_path}: {e}")
        return jsonify({"error": "Could not send file"}), 500

def insert_into_db(document, text_chunk, embedding, university_id=None, school_id=None, chair_id=None, course_id=None):
    try:
        embedding_str = json.dumps(embedding.tolist())
        print(f"Einfügen in DB: Dokument: {document}, Textausschnitt: {text_chunk[:75]}..., Embedding-Länge: {len(embedding)}")
        
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("""
            INSERT INTO embeddings (document, text_chunk, embedding, university_id, school_id, chair_id, course_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (document, text_chunk, embedding_str, university_id, school_id, chair_id, course_id))
        conn.commit()
    except Exception as e:
        print(f"Fehler beim Einfügen in DB: {e}")
    finally:
        conn.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        university_id = request.form.get('university_id')
        school_id = request.form.get('school_id')
        chair_id = request.form.get('chair_id')
        course_id = request.form.get('course_id')

        if not university_id or not school_id or not chair_id or not course_id:
            return "Fehlende Auswahlkriterien.", 400

        # Rest des Codes bleibt gleich
        base_path = app.config['UPLOAD_FOLDER']
        university_path = os.path.join(base_path, university_id if university_id else "")
        school_path = os.path.join(university_path, school_id if school_id else "")
        chair_path = os.path.join(school_path, chair_id if chair_id else "")
        course_path = os.path.join(chair_path, course_id if course_id else "")

        os.makedirs(course_path, exist_ok=True)

        file = request.files['file']
        file_filename = file.filename
        file_path = os.path.join(course_path, file_filename)
        file.save(file_path)

        text = extract_text_from_pdf(file_path)
        sentence_chunks = split_text_into_hierarchical_chunks(text)
        if not sentence_chunks:
            return "No valid sentence chunks found.", 400

        embeddings = generate_embeddings(sentence_chunks)

        for chunk, (index, embedding) in zip(sentence_chunks, embeddings):
            insert_into_db(file_filename, chunk, embedding, university_id, school_id, chair_id, course_id)

        return redirect(url_for('chaten'))

    return render_template('hauptseite.html', embeddings=None)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)