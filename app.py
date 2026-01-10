"""
Malayalam Handwriting Data Collection Tool
Captures Wacom tablet input and saves in sketch-rnn format
"""

import json
import os
import random
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# Data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Corpus directory
CORPUS_DIR = Path(__file__).parent / "corpus"
CORPUS_FILE = CORPUS_DIR / "malayalam_words.json"


def load_corpus_words():
    """Load words from downloaded corpus."""
    if CORPUS_FILE.exists():
        with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
            return corpus.get('categories', {}), corpus.get('all_words', [])
    return {}, []


# Load corpus words
CORPUS_CATEGORIES, ALL_CORPUS_WORDS = load_corpus_words()

# Malayalam words for data collection
# Combines corpus words with specific linguistic categories
MALAYALAM_WORDS = {
    # Words from corpus (dynamically loaded)
    "short": CORPUS_CATEGORIES.get("short", []),       # 1-2 chars from corpus
    "simple": CORPUS_CATEGORIES.get("simple", []),     # 3-4 chars from corpus
    "medium": CORPUS_CATEGORIES.get("medium", []),     # 5-7 chars from corpus
    "long": CORPUS_CATEGORIES.get("long", []),         # 8-10 chars from corpus
    "complex": CORPUS_CATEGORIES.get("complex", []),   # 11+ chars from corpus
    "characters": [
        # Vowels
        "അ", "ആ", "ഇ", "ഈ", "ഉ", "ഊ", "ഋ",
        "എ", "ഏ", "ഐ", "ഒ", "ഓ", "ഔ",
        # Consonants
        "ക", "ഖ", "ഗ", "ഘ", "ങ",
        "ച", "ഛ", "ജ", "ഝ", "ഞ",
        "ട", "ഠ", "ഡ", "ഢ", "ണ",
        "ത", "ഥ", "ദ", "ധ", "ന",
        "പ", "ഫ", "ബ", "ഭ", "മ",
        "യ", "ര", "ല", "വ", "ശ",
        "ഷ", "സ", "ഹ", "ള", "ഴ", "റ",
    ],
    "numbers": [
        # Spelled out numbers
        "ഒന്ന്",      # 1 - onnu
        "രണ്ട്",      # 2 - randu
        "മൂന്ന്",     # 3 - moonnu
        "നാല്",       # 4 - naalu
        "അഞ്ച്",      # 5 - anchu
        "ആറ്",        # 6 - aaru
        "ഏഴ്",        # 7 - ezhu
        "എട്ട്",      # 8 - ettu
        "ഒമ്പത്",     # 9 - ompathu
        "പത്ത്",      # 10 - pathu
        "നൂറ്",       # 100 - nooru
        "ആയിരം",      # 1000 - aayiram
    ],
    "conjuncts": [
        # Common doubled consonants (gemination)
        "ക്ക", "ങ്ങ", "ച്ച", "ഞ്ഞ", "ട്ട",
        "ണ്ണ", "ത്ത", "ന്ന", "പ്പ", "മ്മ",
        "യ്യ", "ല്ല", "വ്വ", "ശ്ശ", "സ്സ",
        # Common nasal + stop conjuncts
        "ങ്ക", "ഞ്ച", "ണ്ട", "ന്ത", "ന്ദ",
        "മ്പ", "ന്ധ",
        # Conjuncts with ര (ra)
        "ക്ര", "ഗ്ര", "ത്ര", "ദ്ര", "പ്ര",
        "ബ്ര", "മ്ര", "വ്ര", "ശ്ര", "സ്ര",
        # Conjuncts with ല (la)
        "ക്ല", "ഗ്ല", "പ്ല", "ബ്ല", "ഫ്ല",
        # Conjuncts with യ (ya)
        "ക്യ", "ഖ്യ", "ത്യ", "ദ്യ", "ന്യ",
        "പ്യ", "മ്യ", "വ്യ", "ശ്യ", "സ്യ",
        # Conjuncts with വ (va)
        "ക്വ", "ത്വ", "ദ്വ", "ശ്വ", "സ്വ",
        # Other common conjuncts
        "ക്ത", "ക്ഷ", "ഗ്ന", "ച്ഛ", "ജ്ഞ",
        "ത്സ", "ത്മ", "ത്ന", "ദ്ധ", "ദ്മ",
        "ന്മ", "പ്ത", "ബ്ദ", "ഭ്യ", "ശ്ച",
        "ഷ്ട", "സ്ത", "സ്ഥ", "സ്ന", "സ്മ",
        "ഹ്ന", "ഹ്മ", "ള്ള", "റ്റ",
        # Three consonant conjuncts (common ones)
        "സ്ത്ര", "ന്ത്ര", "ക്ഷ്മ", "ങ്ക്ഷ",
    ],
    "vowel_signs": [
        # Consonant + vowel sign combinations (using ക as base)
        "കാ", "കി", "കീ", "കു", "കൂ",
        "കെ", "കേ", "കൈ", "കൊ", "കോ", "കൗ",
        "കൃ", "കം", "കഃ",
        # Using other common consonants
        "നാ", "നി", "നു", "നെ", "നൊ",
        "മാ", "മി", "മു", "മെ", "മൊ",
        "ലാ", "ലി", "ലു", "ലെ", "ലൊ",
        "രാ", "രി", "രു", "രെ", "രൊ",
    ],
}


def convert_to_sketch_rnn_format(raw_strokes):
    """
    Convert raw stroke data to sketch-rnn format.

    Sketch-RNN format: list of [dx, dy, pen_state]
    - dx, dy: offset from previous point
    - pen_state: 0 = pen down (drawing), 1 = pen up (moving to next stroke), 2 = end

    Input format: list of strokes, each stroke is list of {x, y, pressure, timestamp}
    """
    if not raw_strokes or len(raw_strokes) == 0:
        return []

    sketch_rnn_data = []
    prev_x, prev_y = None, None

    for stroke_idx, stroke in enumerate(raw_strokes):
        if len(stroke) == 0:
            continue

        for point_idx, point in enumerate(stroke):
            x, y = point['x'], point['y']

            if prev_x is None:
                # First point - use absolute coordinates as first delta
                dx, dy = 0, 0
            else:
                dx = x - prev_x
                dy = y - prev_y

            # Determine pen state
            if point_idx == len(stroke) - 1:
                # Last point of stroke
                if stroke_idx == len(raw_strokes) - 1:
                    # Last point of last stroke
                    pen_state = 2  # End of drawing
                else:
                    pen_state = 1  # Pen up (moving to next stroke)
            else:
                pen_state = 0  # Pen down (drawing)

            sketch_rnn_data.append([dx, dy, pen_state])
            prev_x, prev_y = x, y

    return sketch_rnn_data


def convert_to_stroke3_format(raw_strokes):
    """
    Convert to stroke-3 format used by sketch-rnn.
    Each point: [dx, dy, p1, p2, p3] where p1+p2+p3=1
    - p1=1: pen is touching paper
    - p2=1: pen lifted, stroke ended
    - p3=1: drawing ended
    """
    if not raw_strokes or len(raw_strokes) == 0:
        return np.array([]).reshape(0, 5)

    stroke3_data = []
    prev_x, prev_y = None, None

    for stroke_idx, stroke in enumerate(raw_strokes):
        if len(stroke) == 0:
            continue

        for point_idx, point in enumerate(stroke):
            x, y = point['x'], point['y']

            if prev_x is None:
                dx, dy = 0, 0
            else:
                dx = x - prev_x
                dy = y - prev_y

            # Pen state as one-hot
            p1, p2, p3 = 0, 0, 0

            if point_idx == len(stroke) - 1:
                if stroke_idx == len(raw_strokes) - 1:
                    p3 = 1  # End of drawing
                else:
                    p2 = 1  # End of stroke
            else:
                p1 = 1  # Drawing

            stroke3_data.append([dx, dy, p1, p2, p3])
            prev_x, prev_y = x, y

    return np.array(stroke3_data, dtype=np.float32)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/get_word', methods=['GET'])
def get_word():
    """Get a random Malayalam word for writing practice."""
    category = request.args.get('category', 'all')

    if category == 'all':
        all_words = []
        for words in MALAYALAM_WORDS.values():
            all_words.extend(words)
        word = random.choice(all_words)
    elif category in MALAYALAM_WORDS:
        word = random.choice(MALAYALAM_WORDS[category])
    else:
        word = random.choice(MALAYALAM_WORDS['simple'])

    return jsonify({
        'word': word,
        'category': category
    })


@app.route('/api/get_words', methods=['GET'])
def get_words():
    """Get multiple random Malayalam words."""
    count = int(request.args.get('count', 10))
    category = request.args.get('category', 'all')

    if category == 'all':
        all_words = []
        for words in MALAYALAM_WORDS.values():
            all_words.extend(words)
    else:
        all_words = MALAYALAM_WORDS.get(category, MALAYALAM_WORDS['simple'])

    selected = random.sample(all_words, min(count, len(all_words)))
    return jsonify({'words': selected})


@app.route('/api/save_sample', methods=['POST'])
def save_sample():
    """Save a handwriting sample in sketch-rnn format."""
    data = request.json

    word = data.get('word', '')
    raw_strokes = data.get('strokes', [])
    metadata = data.get('metadata', {})

    if not word or not raw_strokes:
        return jsonify({'error': 'Missing word or strokes'}), 400

    # Convert to sketch-rnn formats
    sketch_rnn_simple = convert_to_sketch_rnn_format(raw_strokes)
    sketch_rnn_stroke3 = convert_to_stroke3_format(raw_strokes)

    # Create sample record
    sample = {
        'word': word,
        'timestamp': time.time(),
        'raw_strokes': raw_strokes,  # Original format with pressure, timestamps
        'sketch_rnn': sketch_rnn_simple,  # [dx, dy, pen_state]
        'stroke3': sketch_rnn_stroke3.tolist(),  # [dx, dy, p1, p2, p3]
        'metadata': {
            'canvas_width': metadata.get('canvas_width', 800),
            'canvas_height': metadata.get('canvas_height', 400),
            'device': metadata.get('device', 'unknown'),
            'pressure_supported': metadata.get('pressure_supported', False),
            **metadata
        }
    }

    # Save to file (one file per word, append samples)
    safe_word = word.replace('/', '_').replace('\\', '_')
    word_file = DATA_DIR / f"{safe_word}.jsonl"

    with open(word_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # Also save to master file
    master_file = DATA_DIR / "all_samples.jsonl"
    with open(master_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    return jsonify({
        'success': True,
        'sample_id': f"{safe_word}_{int(sample['timestamp'])}",
        'stroke_count': len(raw_strokes),
        'point_count': len(sketch_rnn_simple)
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get data collection statistics."""
    master_file = DATA_DIR / "all_samples.jsonl"

    if not master_file.exists():
        return jsonify({
            'total_samples': 0,
            'unique_words': 0,
            'word_counts': {}
        })

    word_counts = {}
    total_samples = 0

    with open(master_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    sample = json.loads(line)
                    word = sample.get('word', 'unknown')
                    word_counts[word] = word_counts.get(word, 0) + 1
                    total_samples += 1
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue

    return jsonify({
        'total_samples': total_samples,
        'unique_words': len(word_counts),
        'word_counts': word_counts
    })


@app.route('/api/export', methods=['GET'])
def export_data():
    """Export data in sketch-rnn compatible numpy format."""
    master_file = DATA_DIR / "all_samples.jsonl"

    if not master_file.exists():
        return jsonify({'error': 'No data to export'}), 404

    # Group by word
    word_data = {}

    with open(master_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    sample = json.loads(line)
                    word = sample.get('word', 'unknown')
                    if word not in word_data:
                        word_data[word] = []
                    word_data[word].append(sample['stroke3'])
                except json.JSONDecodeError:
                    continue

    # Save as npz file
    export_file = DATA_DIR / "malayalam_handwriting.npz"

    # Prepare data arrays
    all_strokes = []
    all_labels = []
    label_to_word = {}
    word_to_label = {}

    for idx, (word, strokes_list) in enumerate(word_data.items()):
        label_to_word[idx] = word
        word_to_label[word] = idx
        for strokes in strokes_list:
            all_strokes.append(np.array(strokes, dtype=np.float32))
            all_labels.append(idx)

    np.savez(
        export_file,
        strokes=np.array(all_strokes, dtype=object),
        labels=np.array(all_labels, dtype=np.int32),
        label_to_word=label_to_word,
        word_to_label=word_to_label
    )

    return jsonify({
        'success': True,
        'file': str(export_file),
        'num_samples': len(all_strokes),
        'num_classes': len(word_data)
    })


@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get available word categories."""
    return jsonify({
        'categories': list(MALAYALAM_WORDS.keys()),
        'counts': {k: len(v) for k, v in MALAYALAM_WORDS.items()},
        'total_words': sum(len(v) for v in MALAYALAM_WORDS.values()),
        'corpus_loaded': len(ALL_CORPUS_WORDS) > 0,
        'corpus_words': len(ALL_CORPUS_WORDS)
    })


@app.route('/api/refresh_corpus', methods=['POST'])
def refresh_corpus():
    """Download fresh corpus from Wikipedia."""
    global CORPUS_CATEGORIES, ALL_CORPUS_WORDS, MALAYALAM_WORDS

    try:
        from corpus_loader import download_corpus

        num_titles = request.json.get('num_titles', 1000) if request.json else 1000
        num_articles = request.json.get('num_articles', 50) if request.json else 50

        corpus = download_corpus(num_titles, num_articles)

        # Reload into memory
        CORPUS_CATEGORIES = corpus.get('categories', {})
        ALL_CORPUS_WORDS = corpus.get('all_words', [])

        # Update MALAYALAM_WORDS
        MALAYALAM_WORDS['short'] = CORPUS_CATEGORIES.get('short', [])
        MALAYALAM_WORDS['simple'] = CORPUS_CATEGORIES.get('simple', [])
        MALAYALAM_WORDS['medium'] = CORPUS_CATEGORIES.get('medium', [])
        MALAYALAM_WORDS['long'] = CORPUS_CATEGORIES.get('long', [])
        MALAYALAM_WORDS['complex'] = CORPUS_CATEGORIES.get('complex', [])

        return jsonify({
            'success': True,
            'total_words': len(ALL_CORPUS_WORDS),
            'categories': {k: len(v) for k, v in CORPUS_CATEGORIES.items()}
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/corpus_stats', methods=['GET'])
def corpus_stats():
    """Get corpus statistics."""
    if CORPUS_FILE.exists():
        with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
            return jsonify({
                'loaded': True,
                'metadata': corpus.get('metadata', {}),
                'categories': {k: len(v) for k, v in corpus.get('categories', {}).items()},
                'total_words': corpus.get('metadata', {}).get('total_words', 0)
            })
    return jsonify({'loaded': False, 'message': 'Corpus not downloaded yet'})


if __name__ == '__main__':
    print("Malayalam Handwriting Data Collection Tool")
    print("Open http://localhost:5000 in your browser")
    print("Make sure your Wacom tablet is connected!")
    app.run(debug=True, host='0.0.0.0', port=5000)
