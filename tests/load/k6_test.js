/**
 * k6 Load Testing Script for MiniCrit API
 *
 * Run with:
 *   # Start the API server first
 *   python -m src.api
 *
 *   # Run load test
 *   k6 run tests/load/k6_test.js
 *
 *   # Run with options
 *   k6 run --vus 10 --duration 60s tests/load/k6_test.js
 *
 *   # Run with environment variables
 *   BASE_URL=http://localhost:8000 k6 run tests/load/k6_test.js
 *
 * Antagon Inc. | CAGE: 17E75
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const critiqueDuration = new Trend('critique_duration');
const batchDuration = new Trend('batch_duration');
const healthCheckDuration = new Trend('health_check_duration');
const tokensGenerated = new Counter('tokens_generated');

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// Test scenarios
export const options = {
  scenarios: {
    // Smoke test - verify basic functionality
    smoke: {
      executor: 'constant-vus',
      vus: 1,
      duration: '10s',
      gracefulStop: '5s',
      tags: { scenario: 'smoke' },
      exec: 'smokeTest',
    },
    // Load test - typical expected load
    load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 5 },   // Ramp up
        { duration: '1m', target: 5 },    // Stay at 5 users
        { duration: '30s', target: 10 },  // Ramp up more
        { duration: '1m', target: 10 },   // Stay at 10 users
        { duration: '30s', target: 0 },   // Ramp down
      ],
      gracefulRampDown: '10s',
      tags: { scenario: 'load' },
      exec: 'loadTest',
      startTime: '15s', // Start after smoke test
    },
    // Stress test - find breaking point
    stress: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 10 },
        { duration: '1m', target: 20 },
        { duration: '1m', target: 30 },
        { duration: '30s', target: 0 },
      ],
      gracefulRampDown: '10s',
      tags: { scenario: 'stress' },
      exec: 'stressTest',
      startTime: '4m', // Start after load test
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% of requests under 2s
    errors: ['rate<0.1'],               // Error rate under 10%
    health_check_duration: ['p(95)<100'], // Health checks under 100ms
    critique_duration: ['p(95)<5000'],    // Critiques under 5s
  },
};

// Sample rationales for testing
const RATIONALES = [
  "AAPL stock will rise because iPhone sales increased 15% last quarter, indicating strong consumer demand.",
  "The company should expand into Europe because competitors have shown 20% revenue growth there.",
  "Bitcoin will reach $100k because institutional adoption is increasing and supply is limited.",
  "We should hire 50 more engineers because our deployment velocity has decreased 30%.",
  "Interest rates will decrease because inflation has dropped 2% and the Fed typically responds.",
  "Our marketing campaign succeeded because sales increased during the campaign period.",
  "The system is secure because no breaches have occurred in the past year.",
  "We should use microservices because Netflix uses them successfully.",
  "The model will generalize well because it achieved 99% accuracy on the test set.",
];

// Helper function to get random rationale
function getRandomRationale() {
  return RATIONALES[Math.floor(Math.random() * RATIONALES.length)];
}

// Health check function
function checkHealth() {
  const start = Date.now();
  const res = http.get(`${BASE_URL}/health`);
  healthCheckDuration.add(Date.now() - start);

  const success = check(res, {
    'health status is 200': (r) => r.status === 200,
    'health response has status': (r) => JSON.parse(r.body).status === 'healthy',
  });

  errorRate.add(!success);
  return success;
}

// Get stats function
function getStats() {
  const res = http.get(`${BASE_URL}/stats`);

  const success = check(res, {
    'stats status is 200': (r) => r.status === 200,
    'stats has request_count': (r) => JSON.parse(r.body).request_count !== undefined,
  });

  errorRate.add(!success);
  return success;
}

// Get metrics function
function getMetrics() {
  const res = http.get(`${BASE_URL}/metrics`);

  const success = check(res, {
    'metrics status is 200': (r) => r.status === 200,
    'metrics contains minicrit_': (r) => r.body.includes('minicrit_'),
  });

  errorRate.add(!success);
  return success;
}

// Create critique function
function createCritique(rationale, maxTokens = 128) {
  const payload = JSON.stringify({
    rationale: rationale,
    max_tokens: maxTokens,
    temperature: 0.7,
  });

  const params = {
    headers: { 'Content-Type': 'application/json' },
  };

  const start = Date.now();
  const res = http.post(`${BASE_URL}/critique`, payload, params);
  critiqueDuration.add(Date.now() - start);

  const success = check(res, {
    'critique status is 200': (r) => r.status === 200,
    'critique has content': (r) => {
      if (r.status === 200) {
        const body = JSON.parse(r.body);
        return body.critique && body.critique.length > 0;
      }
      return false;
    },
  });

  if (res.status === 200) {
    const body = JSON.parse(res.body);
    tokensGenerated.add(body.tokens_generated || 0);
  }

  errorRate.add(!success);
  return success;
}

// Create batch critique function
function createBatchCritique(count = 3) {
  const rationales = [];
  for (let i = 0; i < count; i++) {
    rationales.push(getRandomRationale());
  }

  const payload = JSON.stringify({
    rationales: rationales,
    max_tokens: 64,
    temperature: 0.7,
  });

  const params = {
    headers: { 'Content-Type': 'application/json' },
  };

  const start = Date.now();
  const res = http.post(`${BASE_URL}/critique/batch`, payload, params);
  batchDuration.add(Date.now() - start);

  const success = check(res, {
    'batch status is 200': (r) => r.status === 200,
    'batch has correct count': (r) => {
      if (r.status === 200) {
        const body = JSON.parse(r.body);
        return body.critiques && body.critiques.length === count;
      }
      return false;
    },
  });

  errorRate.add(!success);
  return success;
}

// Smoke test scenario
export function smokeTest() {
  group('smoke_health', () => {
    checkHealth();
  });

  group('smoke_stats', () => {
    getStats();
  });

  group('smoke_metrics', () => {
    getMetrics();
  });

  sleep(1);
}

// Load test scenario
export function loadTest() {
  group('load_health', () => {
    checkHealth();
  });

  group('load_critique', () => {
    createCritique(getRandomRationale());
  });

  group('load_stats', () => {
    getStats();
  });

  sleep(Math.random() * 2 + 1); // 1-3 seconds
}

// Stress test scenario
export function stressTest() {
  const choice = Math.random();

  if (choice < 0.1) {
    group('stress_health', () => {
      checkHealth();
    });
  } else if (choice < 0.3) {
    group('stress_batch', () => {
      createBatchCritique(Math.floor(Math.random() * 5) + 1);
    });
  } else {
    group('stress_critique', () => {
      createCritique(getRandomRationale(), 64);
    });
  }

  sleep(Math.random() * 0.5); // 0-0.5 seconds
}

// Default function (used when running without scenarios)
export default function() {
  loadTest();
}

// Setup function - runs once per VU
export function setup() {
  // Verify server is up
  const res = http.get(`${BASE_URL}/health`);
  if (res.status !== 200) {
    throw new Error(`Server not healthy: ${res.status}`);
  }
  console.log('Server is healthy, starting load test...');
}

// Teardown function - runs once at end
export function teardown(data) {
  console.log('Load test complete');
}
