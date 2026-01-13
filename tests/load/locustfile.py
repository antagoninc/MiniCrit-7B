"""Locust load testing for MiniCrit API.

Run with:
    # Start the API server first
    python -m src.api

    # Run load tests (web UI)
    locust -f tests/load/locustfile.py --host http://localhost:8000

    # Run headless (CI)
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
        --headless -u 10 -r 2 -t 60s --csv=results

Antagon Inc. | CAGE: 17E75
"""

import random
import string
from locust import HttpUser, task, between, tag, events
from locust.runners import MasterRunner


# Sample rationales for testing
SAMPLE_RATIONALES = [
    "AAPL stock will rise because iPhone sales increased 15% last quarter, indicating strong consumer demand and potential for continued growth in the smartphone market.",
    "The company should expand into the European market because competitors have shown 20% revenue growth there, suggesting untapped potential for our products.",
    "Bitcoin will reach $100k because institutional adoption is increasing and supply is limited, creating upward price pressure through basic supply-demand dynamics.",
    "We should hire 50 more engineers because our deployment velocity has decreased 30%, and more developers will linearly increase our shipping speed.",
    "The patient likely has condition X because they present with symptom A and symptom B, which are the two most common indicators according to medical literature.",
    "Interest rates will decrease next quarter because inflation has dropped 2% and the Fed typically responds to lower inflation with rate cuts within 90 days.",
    "Our marketing campaign succeeded because sales increased during the campaign period, demonstrating clear causation between advertising spend and revenue.",
    "The system is secure because no breaches have occurred in the past year, proving that our current security measures are sufficient for protection.",
    "We should use microservices because Netflix uses them successfully, and their architecture choices are applicable to our 10-person startup.",
    "The model will generalize well because it achieved 99% accuracy on the test set, indicating robust performance on unseen data.",
]

# Domains for testing
DOMAINS = ["trading", "compliance", "medical", "general", "statistical"]


def generate_random_rationale(min_len: int = 50, max_len: int = 200) -> str:
    """Generate a random rationale for testing."""
    words = [
        "the", "stock", "will", "rise", "because", "market", "trends", "indicate",
        "growth", "potential", "based", "on", "analysis", "data", "suggests",
        "therefore", "conclude", "evidence", "shows", "reasoning", "logic",
        "implies", "pattern", "historical", "performance", "metrics", "indicate",
    ]
    length = random.randint(min_len, max_len)
    return " ".join(random.choices(words, k=length // 5))


class MiniCritUser(HttpUser):
    """Simulated user for MiniCrit API load testing."""

    # Wait between 1-3 seconds between tasks
    wait_time = between(1, 3)

    def on_start(self):
        """Called when a simulated user starts."""
        # Verify server is healthy
        response = self.client.get("/health")
        if response.status_code != 200:
            raise Exception(f"Server not healthy: {response.status_code}")

    @tag("health")
    @task(1)
    def health_check(self):
        """Test health endpoint."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @tag("stats")
    @task(1)
    def get_stats(self):
        """Test stats endpoint."""
        with self.client.get("/stats", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Stats failed: {response.status_code}")

    @tag("metrics")
    @task(1)
    def get_metrics(self):
        """Test Prometheus metrics endpoint."""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                if "minicrit_" in response.text:
                    response.success()
                else:
                    response.failure("Metrics missing expected content")
            else:
                response.failure(f"Metrics failed: {response.status_code}")

    @tag("critique", "inference")
    @task(5)
    def create_critique_sample(self):
        """Test critique endpoint with sample rationale."""
        rationale = random.choice(SAMPLE_RATIONALES)

        with self.client.post(
            "/critique",
            json={
                "rationale": rationale,
                "max_tokens": 128,
                "temperature": 0.7,
            },
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "critique" in data and len(data["critique"]) > 0:
                    response.success()
                else:
                    response.failure("Empty critique returned")
            elif response.status_code == 503:
                # Server overloaded or model not loaded
                response.failure("Service unavailable (503)")
            else:
                response.failure(f"Critique failed: {response.status_code}")

    @tag("critique", "inference")
    @task(2)
    def create_critique_random(self):
        """Test critique endpoint with random rationale."""
        rationale = generate_random_rationale()

        with self.client.post(
            "/critique",
            json={
                "rationale": rationale,
                "max_tokens": 64,
                "temperature": 0.5,
            },
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 400:
                # Validation error is expected for some random inputs
                response.success()
            elif response.status_code == 503:
                response.failure("Service unavailable (503)")
            else:
                response.failure(f"Critique failed: {response.status_code}")


class MiniCritHeavyUser(HttpUser):
    """Heavy user for stress testing."""

    wait_time = between(0.5, 1)

    @tag("batch", "inference", "heavy")
    @task
    def batch_critique(self):
        """Test batch critique endpoint."""
        rationales = random.sample(SAMPLE_RATIONALES, min(5, len(SAMPLE_RATIONALES)))

        with self.client.post(
            "/critique/batch",
            json={
                "rationales": rationales,
                "max_tokens": 64,
                "temperature": 0.7,
            },
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if len(data.get("critiques", [])) == len(rationales):
                    response.success()
                else:
                    response.failure("Incomplete batch response")
            elif response.status_code == 503:
                response.failure("Service unavailable (503)")
            else:
                response.failure(f"Batch failed: {response.status_code}")


class MiniCritReadOnlyUser(HttpUser):
    """Read-only user for testing non-inference endpoints."""

    wait_time = between(0.1, 0.5)

    @tag("health")
    @task(5)
    def health_check(self):
        """Rapidly check health endpoint."""
        self.client.get("/health")

    @tag("stats")
    @task(3)
    def get_stats(self):
        """Check stats endpoint."""
        self.client.get("/stats")

    @tag("metrics")
    @task(2)
    def get_metrics(self):
        """Check metrics endpoint."""
        self.client.get("/metrics")


# Event hooks for custom reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    print("=" * 60)
    print("MiniCrit Load Test Starting")
    print(f"Target host: {environment.host}")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test ends."""
    print("=" * 60)
    print("MiniCrit Load Test Complete")
    print("=" * 60)


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, **kwargs):
    """Track individual requests for custom metrics."""
    # Could add custom tracking here
    pass
