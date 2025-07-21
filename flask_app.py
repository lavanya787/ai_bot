from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_cors import CORS
from supabase import create_client, Client
import bcrypt
import uuid
from datetime import datetime
import re

app = Flask(__name__)
CORS(app)
app.secret_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN3cmhjc2Z1b3JqcXN6cW9vaGZiIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MDkxODA3MCwiZXhwIjoyMDY2NDk0MDcwfQ.aYmADn3cpUhPlNSUiKlb-EveEyyE7-8FgYVq7L4A2OA'

# Supabase setup
SUPABASE_URL = "https://swrhcsfuorjqszqoohfb.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN3cmhjc2Z1b3JqcXN6cW9vaGZiIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MDkxODA3MCwiZXhwIjoyMDY2NDk0MDcwfQ.aYmADn3cpUhPlNSUiKlb-EveEyyE7-8FgYVq7L4A2OA"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def is_valid_uuid(val):
    return bool(re.fullmatch(
        r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$', val
    ))

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/auth')
def index():
    return render_template('index.html')

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').lower().strip()
        password = data.get('password', '').strip()

        if not username or not email or not password:
            return jsonify({'message': 'All fields are required'}), 400

        existing_user = supabase.table('users').select('*').eq('email', email).execute()
        if existing_user.data:
            return jsonify({'message': 'User already exists'}), 400

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        user_id = str(uuid.uuid4())
        if not is_valid_uuid(user_id):
            return jsonify({'message': 'Invalid user ID'}), 500

        new_user = {
            'user_id': user_id,
            'username': username,
            'email': email,
            'password': hashed_password,
            'registration_time': datetime.utcnow().isoformat()
        }

        supabase.table('users').insert(new_user).execute()
        return jsonify({'message': 'Registration successful', 'user_id': user_id}), 200

    except Exception as e:
        print(f"[REGISTER ERROR] {str(e)}")
        return jsonify({'message': 'Internal server error', 'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        password = data.get('password', '').strip()

        if not email or not password:
            return jsonify({'message': 'Email and password are required'}), 400

        user_response = supabase.table('users').select('*').eq('email', email).execute()

        if not user_response.data:
            return jsonify({'message': 'User not found'}), 404

        user_data = user_response.data[0]

        if not bcrypt.checkpw(password.encode('utf-8'), user_data['password'].encode('utf-8')):
            return jsonify({'message': 'Incorrect password'}), 401

        login_time = datetime.utcnow().isoformat()
        supabase.table('users').update({'last_login': login_time}).eq('email', email).execute()

        session['user_id'] = user_data['user_id']

        return jsonify({
            'message': 'Login successful!',
            'user_id': user_data['user_id'],
            'redirect': f'http://localhost:8501?user_id={user_data["user_id"]}'
        }), 200

    except Exception as e:
        print(f"[LOGIN ERROR] {str(e)}")
        return jsonify({'message': 'Internal server error', 'error': str(e)}), 500

@app.route('/api/save_query_response', methods=['POST'])
def save_query_response():
    try:
        data = request.get_json()
        print("📥 Received from Streamlit:", data)  # 👈 Terminal print

        user_id = data.get('user_id')
        query_text = data.get('query')
        response_text = data.get('response')

        if not user_id or not query_text or not response_text:
            print("[SAVE] ❌ Missing required fields")
            return jsonify({'message': 'Missing data'}), 400

        # Insert into queries table
        query_insert = supabase.table('queries').insert({
            'user_id': user_id,
            'query_text': query_text,
            'timestamp': datetime.utcnow().isoformat()
        }).execute()
        print(f"[SAVE] ✅ Query insert result: {query_insert}")

        # Get query_id from insert result
        if not query_insert.data:
            print("[SAVE] ❌ Query insert returned no data")
            return jsonify({'message': 'Failed to insert query'}), 500

        query_id = query_insert.data[0].get('query_id') or query_insert.data[0].get('id')
        print(f"[SAVE] ✅ Inserted query_id: {query_id}")

        # Insert into responses table
        response_insert = supabase.table('responses').insert({
            'query_id': query_id,
            'user_id': user_id,
            'response_text': response_text,
            'timestamp': datetime.utcnow().isoformat()
        }).execute()
        print(f"[SAVE] ✅ Response insert result: {response_insert}")

        return jsonify({'message': 'Query and response saved successfully'}), 200

    except Exception as e:
        print(f"[SAVE ERROR] ❌ {str(e)}")
        return jsonify({'message': 'Internal server error', 'error': str(e)}), 500


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('homepage'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)