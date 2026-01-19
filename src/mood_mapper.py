"""
Mood-aware mapping system
Mapping emotional states to book categories and enhancing user queries
"""

class MoodMapper:
    """Mapping moods to preferred book categories and processing hybrid queries"""

    def __init__(self):
        # Initializing mood-to-category mappings
        self.mood_mappings = {
            'anxious': {
                'primary': ['Self-Help', 'Fiction', 'Young Adult'],
                'keywords': 'calming peaceful mindfulness relaxation comfort',
                'avoid': ['Thriller', 'Horror'],
                'reasoning': 'Light and comforting reads supporting anxiety relief'
            },
            'sad': {
                'primary': ['Fiction', 'Biography', 'Self-Help'],
                'keywords': 'uplifting hopeful inspiring heartwarming encouraging',
                'avoid': ['Horror'],
                'reasoning': 'Uplifting stories supporting mood improvement'
            },
            'stressed': {
                'primary': ['Fiction', 'Young Adult', 'General'],
                'keywords': 'escapism light entertaining easy quick',
                'avoid': ['Business', 'Science'],
                'reasoning': 'Easy and engaging reads supporting mental relaxation'
            },
            'bored': {
                'primary': ['Science Fiction', 'Fantasy', 'Mystery'],
                'keywords': 'exciting adventure thrilling engaging captivating',
                'avoid': [],
                'reasoning': 'Fast-paced narratives supporting engagement'
            },
            'lonely': {
                'primary': ['Fiction', 'Biography', 'Young Adult'],
                'keywords': 'friendship connection relationships community warmth',
                'avoid': [],
                'reasoning': 'Stories focusing on human connection'
            },
            'curious': {
                'primary': ['Science', 'History', 'Biography'],
                'keywords': 'learning discovery knowledge facts education',
                'avoid': [],
                'reasoning': 'Educational content supporting curiosity'
            },
            'motivated': {
                'primary': ['Business', 'Self-Help', 'Biography'],
                'keywords': 'success achievement inspiration growth development',
                'avoid': [],
                'reasoning': 'Inspirational content supporting ambition'
            },
            'thoughtful': {
                'primary': ['Fiction', 'Biography', 'History'],
                'keywords': 'philosophical deep meaningful contemplative reflective',
                'avoid': [],
                'reasoning': 'Thought-provoking narratives supporting reflection'
            },
            'creative': {
                'primary': ['Arts', 'Fiction', 'Biography'],
                'keywords': 'artistic imaginative innovative creative inspiration',
                'avoid': [],
                'reasoning': 'Creatively stimulating content'
            },
            'adventurous': {
                'primary': ['Science Fiction', 'Fantasy', 'History'],
                'keywords': 'adventure exploration journey quest discovery',
                'avoid': [],
                'reasoning': 'Exploratory narratives supporting adventure-seeking moods'
            },
            'relaxed': {
                'primary': ['Fiction', 'General', 'Young Adult'],
                'keywords': 'pleasant comfortable enjoyable light entertaining',
                'avoid': ['Thriller', 'Horror'],
                'reasoning': 'Gentle and enjoyable reads'
            },
            'romantic': {
                'primary': ['Fiction', 'Young Adult'],
                'keywords': 'romance love relationship heartwarming emotional',
                'avoid': [],
                'reasoning': 'Romantic and emotional narratives'
            }
        }

        # Initializing mood synonym mappings
        self.mood_synonyms = {
            'worried': 'anxious',
            'nervous': 'anxious',
            'scared': 'anxious',
            'depressed': 'sad',
            'down': 'sad',
            'upset': 'sad',
            'overwhelmed': 'stressed',
            'tired': 'stressed',
            'exhausted': 'stressed',
            'isolated': 'lonely',
            'alone': 'lonely',
            'interested': 'curious',
            'wondering': 'curious',
            'ambitious': 'motivated',
            'driven': 'motivated',
            'inspired': 'motivated',
            'pensive': 'thoughtful',
            'reflective': 'thoughtful',
            'calm': 'relaxed',
            'peaceful': 'relaxed'
        }

    def detect_mood(self, query):
        """Detecting mood indicators within the user query"""
        query_lower = query.lower()
        detected = []

        # Detecting direct mood words
        for mood in self.mood_mappings:
            if mood in query_lower:
                detected.append(mood)

        # Detecting synonyms mapping to moods
        for synonym, mood in self.mood_synonyms.items():
            if synonym in query_lower and mood not in detected:
                detected.append(mood)

        return detected

    def detect_book_reference(self, query):
        """Detecting user references to specific books inside the query"""
        query_lower = query.lower()
        patterns = [
            "i loved", "i liked", "i enjoyed", "similar to",
            "like", "after reading", "finished reading"
        ]

        for pattern in patterns:
            if pattern in query_lower:
                # Extracting quoted text if present
                if '"' in query or "'" in query:
                    import re
                    match = re.search(r'["\']([^"\']+)["\']', query)
                    if match:
                        return match.group(1)
                return "BOOK_REFERENCE_DETECTED"

        return None

    def enhance_query_with_mood(self, query, moods):
        """Enhancing the user query with mood-specific descriptive keywords"""
        if not moods:
            return query

        mood_keywords = [
            self.mood_mappings[m]['keywords']
            for m in moods
            if m in self.mood_mappings
        ]

        return f"{query} {' '.join(mood_keywords)}"

    def get_mood_categories(self, moods):
        """Collecting category preferences and avoid lists based on moods"""
        if not moods:
            return {'primary': [], 'avoid': []}

        primary = []
        avoid = []

        for mood in moods:
            if mood in self.mood_mappings:
                primary.extend(self.mood_mappings[mood]['primary'])
                avoid.extend(self.mood_mappings[mood]['avoid'])

        # Removing duplicates while preserving ordering
        primary = list(dict.fromkeys(primary))
        avoid = list(dict.fromkeys(avoid))

        return {'primary': primary, 'avoid': avoid}

    def process_query(self, query):
        """Processing the complete query to determine mood, preferences, and use case"""
        moods = self.detect_mood(query)
        book_ref = self.detect_book_reference(query)
        categories = self.get_mood_categories(moods)
        enhanced = self.enhance_query_with_mood(query, moods)

        if moods and book_ref:
            use_case = "USE_CASE_3_COMBINED"
        elif moods:
            use_case = "USE_CASE_1_MOOD"
        elif book_ref:
            use_case = "USE_CASE_2_PREFERENCE"
        else:
            use_case = "GENERAL_QUERY"

        return {
            'original_query': query,
            'enhanced_query': enhanced,
            'detected_moods': moods,
            'book_reference': book_ref,
            'use_case': use_case,
            'recommended_categories': categories['primary'],
            'avoid_categories': categories['avoid'],
            'mood_reasoning': [
                self.mood_mappings[m]['reasoning']
                for m in moods if m in self.mood_mappings
            ]
        }

    def explain_recommendation(self, analysis):
        """Generating a human-readable explanation for recommended books"""
        use_case = analysis['use_case']
        moods = analysis['detected_moods']

        if use_case == "USE_CASE_1_MOOD":
            mood_text = ', '.join(moods)
            return f"Based on your mood ({mood_text}), recommending books that are {', '.join(analysis['mood_reasoning'])}"

        if use_case == "USE_CASE_2_PREFERENCE":
            return "Recommending books based on your reading preferences"

        if use_case == "USE_CASE_3_COMBINED":
            mood_text = ', '.join(moods)
            return f"Combining your mood ({mood_text}) with your reading preferences to produce tailored recommendations"

        return "Based on your query, recommending the best matching books"
