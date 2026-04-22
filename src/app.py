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
instance_dir = os.path.join(project_root, 'instance')
os.makedirs(instance_dir, exist_ok=True)
database_path = os.path.join(instance_dir, 'data.db')

# Serve React build files from <project_root>/frontend/dist
app = Flask(__name__,
    static_folder=os.path.join(project_root, 'frontend', 'dist'),
    instance_path=instance_dir,
    static_url_path='')
CORS(app)

# Configure SQLite database with a stable absolute path so startup
# does not depend on the directory you launch Flask from.
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{database_path}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database with app
db.init_app(app)

# Register routes
register_routes(app)

# Function to initialize database, change this to your own database initialization logic
def init_db():
   with app.app_context():
        try:
            db.create_all()
        except Exception:
            pass

        if AitaPost.query.count() == 0:
            try:
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
            except Exception:
                db.session.rollback()
init_db()

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5001)
