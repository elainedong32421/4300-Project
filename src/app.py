import json
import os
import csv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from flask import Flask
from flask_cors import CORS
from models import db, AitaPost
from routes import register_routes

# src/ directory and project root (one level up)
current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_directory)

# Serve React build files from <project_root>/frontend/dist
app = Flask(__name__,
    static_folder=os.path.join(project_root, 'frontend', 'dist'),
    static_url_path='')
CORS(app)

# Configure SQLite database - using 3 slashes for relative path
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database with app
db.init_app(app)

# Register routes
register_routes(app)

# Function to initialize database, change this to your own database initialization logic
def init_db():
   with app.app_context():
        db.create_all()

        if AitaPost.query.count() == 0:
            csv_file_path = os.path.join(project_root, 'data', 'AITA_clean1.csv')

            with open(csv_file_path, 'r', encoding='utf-8', newline='') as file:
                reader = csv.DictReader(file)

                for row in reader:
                    post = AitaPost(
                        id=int(row['id']) if row['id'] else None,
                        submission_id=row['submission_id'],
                        title=row['title'],
                        selftext=row['selftext'],
                        score=int(row['score']) if row['score'] else 0
                    )
                    db.session.add(post)

                db.session.commit()
                print("Loaded AITA CSV into database")
init_db()

# Warm up the search index at startup so first request isn't slow
from routes import _tfidf_index
print("Warming up search index...")
_tfidf_index()
print("Index ready.")

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5001)
