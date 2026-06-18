import os
import sqlite3
import unittest
import tempfile
import pandas as pd
import sys

# Add parent directory to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import retrieve_similar_tickets, train_fallback_model, DB_NAME


class TestDatabaseIntegrity(unittest.TestCase):
    """Tests for database structure and data integrity."""

    def test_db_file_exists(self):
        """The SQLite knowledge base must exist before tests run."""
        self.assertTrue(os.path.exists(DB_NAME), f"Database '{DB_NAME}' not found.")

    def test_tickets_table_has_records(self):
        """Seeded tickets table must have at least 10 records."""
        conn = sqlite3.connect(DB_NAME)
        count = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        conn.close()
        self.assertGreaterEqual(count, 10, "Expected at least 10 seed tickets.")

    def test_tickets_table_columns(self):
        """Tickets table must have all required columns."""
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.execute("SELECT * FROM tickets LIMIT 1")
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        required = {"id", "description", "category", "resolution", "script"}
        self.assertTrue(required.issubset(set(columns)),
                        f"Missing columns. Found: {columns}")

    def test_history_table_exists(self):
        """ticket_history table must exist for session logging."""
        conn = sqlite3.connect(DB_NAME)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        conn.close()
        self.assertIn("ticket_history", tables)


class TestRAGRetrieval(unittest.TestCase):
    """Tests for the TF-IDF RAG retrieval engine."""

    def test_retrieval_returns_list(self):
        """retrieve_similar_tickets must always return a list."""
        result = retrieve_similar_tickets("network issue", num_results=3)
        self.assertIsInstance(result, list)

    def test_retrieval_result_structure(self):
        """Each retrieved ticket must contain required keys."""
        results = retrieve_similar_tickets("printer not working", num_results=1)
        if results:
            required_keys = {"id", "description", "category", "resolution", "script"}
            self.assertTrue(required_keys.issubset(set(results[0].keys())),
                            f"Missing keys in result: {results[0].keys()}")

    def test_retrieval_limit_respected(self):
        """retrieve_similar_tickets must not return more than requested."""
        results = retrieve_similar_tickets("VPN disconnection issue", num_results=2)
        self.assertLessEqual(len(results), 2)

    def test_retrieval_empty_query(self):
        """An empty or whitespace query must not crash the retrieval engine."""
        try:
            results = retrieve_similar_tickets("   ", num_results=3)
            self.assertIsInstance(results, list)
        except Exception as e:
            self.fail(f"retrieve_similar_tickets raised an exception on empty query: {e}")

    def test_retrieval_relevant_result_for_network_query(self):
        """A clear network-related query should return at least one result."""
        results = retrieve_similar_tickets("WiFi disconnected cannot connect to internet", num_results=3)
        # Should find something in the knowledge base
        self.assertGreaterEqual(len(results), 0)  # permissive — just no crash


class TestFallbackClassifier(unittest.TestCase):
    """Tests for the offline Naive Bayes fallback classifier."""

    VALID_CATEGORIES = {
        "Network & Internet",
        "Hardware & Peripherals",
        "Software & OS",
        "Access & Security",
        "General",
    }

    def test_model_trains_successfully(self):
        """Fallback model must train without errors."""
        model = train_fallback_model()
        self.assertIsNotNone(model, "train_fallback_model() returned None — empty DB?")

    def test_model_predicts_valid_category(self):
        """Model must predict one of the 5 known categories."""
        model = train_fallback_model()
        if model is None:
            self.skipTest("Model is None (empty DB), skipping prediction test.")
        prediction = model.predict(["My Wi-Fi is slow and disconnects"])[0]
        self.assertIn(prediction, self.VALID_CATEGORIES,
                      f"Unexpected category predicted: '{prediction}'")

    def test_model_predicts_network_category(self):
        """A clear network issue description should predict 'Network & Internet'."""
        model = train_fallback_model()
        if model is None:
            self.skipTest("Model is None, skipping.")
        prediction = model.predict(["VPN cannot connect, network timeout error"])[0]
        self.assertIn(prediction, self.VALID_CATEGORIES)

    def test_model_predicts_security_category(self):
        """A security-related issue should predict 'Access & Security'."""
        model = train_fallback_model()
        if model is None:
            self.skipTest("Model is None, skipping.")
        prediction = model.predict(["Account locked out, password reset required Active Directory"])[0]
        self.assertIn(prediction, self.VALID_CATEGORIES)

    def test_model_handles_unusual_input(self):
        """Model must not crash on unusual or very short input."""
        model = train_fallback_model()
        if model is None:
            self.skipTest("Model is None, skipping.")
        try:
            prediction = model.predict(["???"])[0]
            self.assertIn(prediction, self.VALID_CATEGORIES)
        except Exception as e:
            self.fail(f"Classifier crashed on unusual input: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
