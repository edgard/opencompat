#!/usr/bin/env python3
"""
OpenCompat E2E Compatibility Test Suite

A comprehensive test suite to validate OpenAI API compatibility.

Usage:
    python tests/e2e.py                        # Run all tests (chatgpt provider)
    python tests/e2e.py --provider copilot     # Run with copilot provider
    python tests/e2e.py --provider chatgpt     # Run with chatgpt provider
    python tests/e2e.py connectivity           # Run specific category
    python tests/e2e.py --test single_turn     # Run single test
    python tests/e2e.py --verbose              # Verbose output
    python tests/e2e.py --server URL           # Custom server URL
    python tests/e2e.py --timeout 60           # Custom timeout
    python tests/e2e.py --list                 # List all tests
    python tests/e2e.py --json                 # JSON output
"""

import argparse
import json
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# --- Exceptions ---


class TestAssertionError(Exception):
    """Raised when a test assertion fails."""

    def __init__(self, message: str, expected: Any = None, got: Any = None):
        self.message = message
        self.expected = expected
        self.got = got
        super().__init__(message)


class SkipTest(Exception):
    """Raised to skip a test."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class TestTimeout(Exception):
    """Raised when a test times out."""

    pass


# --- Data Classes ---


@dataclass
class TestResult:
    """Result of a single test execution."""

    name: str
    category: str
    passed: bool
    skipped: bool
    duration: float
    error: Optional[str] = None
    expected: Optional[Any] = None
    got: Optional[Any] = None
    skip_reason: Optional[str] = None


@dataclass
class TestInfo:
    """Information about a registered test."""

    name: str
    category: str
    func: Callable


# --- Test Suite ---


class TestSuite:
    """Main test suite runner."""

    # Provider configurations
    # ChatGPT uses gpt-5 (reasoning model), Copilot uses gpt-4o (standard model)
    # gpt-5 on Copilot has limitations (no stop, max_tokens, specific tool_choice)
    PROVIDERS = {
        "chatgpt": {
            "default_model": "chatgpt/gpt-5",
            "expected_models": [
                "chatgpt/gpt-5",
            ],
            "supports_effort_suffix": True,
            "supports_reasoning_headers": True,
            "supports_sampling_params": False,  # ChatGPT ignores temperature, top_p, etc.
        },
        "copilot": {
            "default_model": "copilot/gpt-4o",
            "expected_models": [
                "copilot/gpt-4o",
            ],
            "supports_effort_suffix": False,
            "supports_reasoning_headers": False,
            "supports_sampling_params": True,  # Copilot supports temperature, top_p, etc.
        },
    }

    def __init__(self, base_url: str, timeout: int, verbose: bool, provider: str, model: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verbose = verbose
        self.console = Console()

        # Provider configuration
        self.provider = provider
        self.provider_config = self.PROVIDERS[provider]
        self.model = model if model else self.provider_config["default_model"]

        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="not-needed",
            timeout=timeout,
        )

        # Test registry
        self.tests: List[TestInfo] = []
        self.results: List[TestResult] = []

        # Categories in display order
        self.category_order = [
            "connectivity",
            "basic_chat",
            "streaming",
            "models",
            "tools",
            "parameters",
            "errors",
            "response_format",
        ]

    def test(self, name: str, category: str) -> Callable:
        """Decorator to register a test."""

        def decorator(func: Callable) -> Callable:
            self.tests.append(TestInfo(name=name, category=category, func=func))
            return func

        return decorator

    # --- Assertions ---

    def assert_equal(self, actual: Any, expected: Any, msg: str) -> None:
        """Assert two values are equal."""
        if actual != expected:
            raise TestAssertionError(msg, expected=expected, got=actual)

    def assert_true(self, condition: bool, msg: str) -> None:
        """Assert condition is true."""
        if not condition:
            raise TestAssertionError(msg)

    def assert_false(self, condition: bool, msg: str) -> None:
        """Assert condition is false."""
        if condition:
            raise TestAssertionError(msg)

    def assert_contains(self, haystack: str, needle: str, msg: str) -> None:
        """Assert string contains substring."""
        if needle not in haystack:
            raise TestAssertionError(msg, expected=f"contains '{needle}'", got=haystack)

    def assert_not_contains(self, haystack: str, needle: str, msg: str) -> None:
        """Assert string does not contain substring."""
        if needle in haystack:
            raise TestAssertionError(
                msg, expected=f"does not contain '{needle}'", got=haystack
            )

    def assert_in(self, item: Any, collection: Any, msg: str) -> None:
        """Assert item is in collection."""
        if item not in collection:
            raise TestAssertionError(msg, expected=f"'{item}' in collection", got=collection)

    def assert_not_in(self, item: Any, collection: Any, msg: str) -> None:
        """Assert item is not in collection."""
        if item in collection:
            raise TestAssertionError(
                msg, expected=f"'{item}' not in collection", got=collection
            )

    def assert_is_none(self, obj: Any, msg: str) -> None:
        """Assert object is None."""
        if obj is not None:
            raise TestAssertionError(msg, expected=None, got=obj)

    def assert_is_not_none(self, obj: Any, msg: str) -> None:
        """Assert object is not None."""
        if obj is None:
            raise TestAssertionError(msg, expected="not None", got=None)

    def assert_greater(self, a: Any, b: Any, msg: str) -> None:
        """Assert a > b."""
        if not a > b:
            raise TestAssertionError(msg, expected=f"> {b}", got=a)

    def assert_greater_equal(self, a: Any, b: Any, msg: str) -> None:
        """Assert a >= b."""
        if not a >= b:
            raise TestAssertionError(msg, expected=f">= {b}", got=a)

    def assert_less(self, a: Any, b: Any, msg: str) -> None:
        """Assert a < b."""
        if not a < b:
            raise TestAssertionError(msg, expected=f"< {b}", got=a)

    def assert_type(self, obj: Any, expected_type: type, msg: str) -> None:
        """Assert object is of expected type."""
        if not isinstance(obj, expected_type):
            raise TestAssertionError(
                msg, expected=expected_type.__name__, got=type(obj).__name__
            )

    def assert_has_key(self, obj: dict, key: str, msg: str) -> None:
        """Assert dict has key."""
        if not isinstance(obj, dict):
            raise TestAssertionError(msg, expected="dict", got=type(obj).__name__)
        if key not in obj:
            raise TestAssertionError(msg, expected=f"key '{key}'", got=list(obj.keys()))

    def assert_has_attr(self, obj: Any, attr: str, msg: str) -> None:
        """Assert object has attribute."""
        if not hasattr(obj, attr):
            raise TestAssertionError(msg, expected=f"attribute '{attr}'", got=dir(obj))

    def assert_status_code(self, response: requests.Response, expected: int, msg: str) -> None:
        """Assert HTTP response status code."""
        if response.status_code != expected:
            raise TestAssertionError(msg, expected=expected, got=response.status_code)

    # --- Skip ---

    def skip(self, reason: str) -> None:
        """Skip current test."""
        raise SkipTest(reason)

    def skip_if(self, condition: bool, reason: str) -> None:
        """Skip current test if condition is true."""
        if condition:
            raise SkipTest(reason)

    # --- Execution ---

    def _run_single_test(self, test_info: TestInfo) -> TestResult:
        """Run a single test and return result."""
        start_time = time.time()

        # Set up timeout handler
        def timeout_handler(signum, frame):
            raise TestTimeout(f"Test timed out after {self.timeout}s")

        # Only use signal on Unix-like systems
        old_handler = None
        if hasattr(signal, "SIGALRM"):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)

        try:
            test_info.func(self)
            duration = time.time() - start_time
            return TestResult(
                name=test_info.name,
                category=test_info.category,
                passed=True,
                skipped=False,
                duration=duration,
            )
        except SkipTest as e:
            duration = time.time() - start_time
            return TestResult(
                name=test_info.name,
                category=test_info.category,
                passed=True,
                skipped=True,
                duration=duration,
                skip_reason=e.reason,
            )
        except TestAssertionError as e:
            duration = time.time() - start_time
            return TestResult(
                name=test_info.name,
                category=test_info.category,
                passed=False,
                skipped=False,
                duration=duration,
                error=e.message,
                expected=e.expected,
                got=e.got,
            )
        except TestTimeout as e:
            duration = time.time() - start_time
            return TestResult(
                name=test_info.name,
                category=test_info.category,
                passed=False,
                skipped=False,
                duration=duration,
                error=str(e),
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name=test_info.name,
                category=test_info.category,
                passed=False,
                skipped=False,
                duration=duration,
                error=f"{type(e).__name__}: {e}",
            )
        finally:
            if hasattr(signal, "SIGALRM") and old_handler is not None:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    def run(
        self,
        categories: Optional[List[str]] = None,
        test_names: Optional[List[str]] = None,
    ) -> None:
        """Run tests, optionally filtered by category or test name."""
        self.results = []

        # Filter tests
        tests_to_run = self.tests
        if test_names:
            tests_to_run = [t for t in tests_to_run if t.name in test_names]
        elif categories:
            tests_to_run = [t for t in tests_to_run if t.category in categories]

        if not tests_to_run:
            self.console.print("[yellow]No tests to run[/yellow]")
            return

        # Print header
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]OpenCompat E2E Test Suite[/bold]\n"
                f"Server: {self.base_url}\n"
                f"Provider: {self.provider} (model: {self.model})",
                expand=False,
            )
        )
        self.console.print()

        # Group by category
        categories_seen = []
        tests_by_category: Dict[str, List[TestInfo]] = {}
        for test in tests_to_run:
            if test.category not in tests_by_category:
                tests_by_category[test.category] = []
                categories_seen.append(test.category)
            tests_by_category[test.category].append(test)

        # Sort categories by predefined order
        def category_sort_key(cat):
            try:
                return self.category_order.index(cat)
            except ValueError:
                return len(self.category_order)

        categories_seen.sort(key=category_sort_key)

        # Run tests
        for category in categories_seen:
            self.console.print(f"[bold cyan]{category}[/bold cyan]")

            for test_info in tests_by_category[category]:
                result = self._run_single_test(test_info)
                self.results.append(result)

                # Print result
                if result.skipped:
                    status = "[yellow]○[/yellow]"
                    suffix = f"[dim]skip[/dim]"
                elif result.passed:
                    status = "[green]✓[/green]"
                    suffix = ""
                else:
                    status = "[red]✗[/red]"
                    suffix = ""

                duration_str = f"{result.duration:.2f}s"
                name_padded = result.name.ljust(50)

                if result.skipped:
                    self.console.print(f"  {status} {name_padded} {suffix}")
                    self.console.print(f"    [dim]│ Reason: {result.skip_reason}[/dim]")
                elif result.passed:
                    self.console.print(f"  {status} {name_padded} [dim]{duration_str}[/dim]")
                else:
                    self.console.print(f"  {status} {name_padded} [dim]{duration_str}[/dim]")
                    self.console.print(f"    [red]│ {result.error}[/red]")
                    if result.expected is not None:
                        self.console.print(f"    [dim]│ Expected: {result.expected}[/dim]")
                    if result.got is not None:
                        got_str = str(result.got)
                        if len(got_str) > 100:
                            got_str = got_str[:100] + "..."
                        self.console.print(f"    [dim]│ Got: {got_str}[/dim]")

            self.console.print()

    def print_summary(self) -> None:
        """Print test summary."""
        passed = sum(1 for r in self.results if r.passed and not r.skipped)
        failed = sum(1 for r in self.results if not r.passed)
        skipped = sum(1 for r in self.results if r.skipped)
        total_duration = sum(r.duration for r in self.results)

        summary_parts = []
        if passed > 0:
            summary_parts.append(f"[green]{passed} passed[/green]")
        if failed > 0:
            summary_parts.append(f"[red]{failed} failed[/red]")
        if skipped > 0:
            summary_parts.append(f"[yellow]{skipped} skipped[/yellow]")

        summary = ", ".join(summary_parts)

        self.console.print(
            Panel(
                f"Results: {summary}\nDuration: {total_duration:.2f}s",
                expand=False,
            )
        )

    def results_as_json(self) -> str:
        """Return results as JSON string."""
        return json.dumps(
            {
                "server": self.base_url,
                "provider": self.provider,
                "model": self.model,
                "results": [
                    {
                        "name": r.name,
                        "category": r.category,
                        "passed": r.passed,
                        "skipped": r.skipped,
                        "duration": r.duration,
                        "error": r.error,
                        "skip_reason": r.skip_reason,
                    }
                    for r in self.results
                ],
                "summary": {
                    "passed": sum(1 for r in self.results if r.passed and not r.skipped),
                    "failed": sum(1 for r in self.results if not r.passed),
                    "skipped": sum(1 for r in self.results if r.skipped),
                    "total_duration": sum(r.duration for r in self.results),
                },
            },
            indent=2,
        )

    def list_tests(self) -> None:
        """List all available tests."""
        table = Table(title="Available Tests")
        table.add_column("Category", style="cyan")
        table.add_column("Test Name", style="white")

        # Group by category
        tests_by_category: Dict[str, List[str]] = {}
        for test in self.tests:
            if test.category not in tests_by_category:
                tests_by_category[test.category] = []
            tests_by_category[test.category].append(test.name)

        # Sort categories
        def category_sort_key(cat):
            try:
                return self.category_order.index(cat)
            except ValueError:
                return len(self.category_order)

        sorted_categories = sorted(tests_by_category.keys(), key=category_sort_key)

        for category in sorted_categories:
            for i, test_name in enumerate(sorted(tests_by_category[category])):
                cat_display = category if i == 0 else ""
                table.add_row(cat_display, test_name)

        self.console.print(table)

    def get_failure_count(self) -> int:
        """Return number of failed tests."""
        return sum(1 for r in self.results if not r.passed and not r.skipped)


# --- Test Registration ---


def register_tests(suite: TestSuite) -> None:
    """Register all tests with the suite."""

    # ==========================================================================
    # CONNECTIVITY TESTS
    # ==========================================================================

    @suite.test("health_endpoint", "connectivity")
    def _(s: TestSuite):
        """GET /health returns 200 with status ok."""
        r = requests.get(f"{s.base_url}/health", timeout=s.timeout)
        s.assert_status_code(r, 200, "Health endpoint should return 200")
        data = r.json()
        s.assert_equal(data.get("status"), "ok", "Status should be 'ok'")

    @suite.test("health_method_not_allowed", "connectivity")
    def _(s: TestSuite):
        """POST /health returns 405."""
        r = requests.post(f"{s.base_url}/health", timeout=s.timeout)
        s.assert_status_code(r, 405, "POST /health should return 405")

    @suite.test("models_list", "connectivity")
    def _(s: TestSuite):
        """GET /v1/models returns models list."""
        models = s.client.models.list()
        s.assert_is_not_none(models.data, "Models data should not be None")
        s.assert_greater(len(models.data), 0, "Should have at least one model")

    @suite.test("models_structure", "connectivity")
    def _(s: TestSuite):
        """Each model has required fields."""
        models = s.client.models.list()
        for model in models.data:
            s.assert_has_attr(model, "id", "Model should have 'id'")
            s.assert_has_attr(model, "object", "Model should have 'object'")
            s.assert_has_attr(model, "created", "Model should have 'created'")
            s.assert_has_attr(model, "owned_by", "Model should have 'owned_by'")
            s.assert_equal(model.object, "model", "Model object should be 'model'")

    @suite.test("models_expected", "connectivity")
    def _(s: TestSuite):
        """Expected models are present for the selected provider."""
        models = s.client.models.list()
        model_ids = [m.id for m in models.data]

        # Check expected models for the provider
        expected = s.provider_config["expected_models"]
        for exp in expected:
            s.assert_in(exp, model_ids, f"Expected model '{exp}' in list")

    # ==========================================================================
    # BASIC CHAT TESTS
    # ==========================================================================

    @suite.test("single_turn", "basic_chat")
    def _(s: TestSuite):
        """Basic single-turn chat works."""
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Say exactly 'hello' and nothing else."}],
        )
        s.assert_is_not_none(r.choices, "Should have choices")
        s.assert_greater(len(r.choices), 0, "Should have at least one choice")
        content = r.choices[0].message.content.lower()
        s.assert_contains(content, "hello", "Response should contain 'hello'")

    @suite.test("response_structure", "basic_chat")
    def _(s: TestSuite):
        """Response has all required fields."""
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Hi"}],
        )
        s.assert_is_not_none(r.id, "Response should have 'id'")
        s.assert_equal(r.object, "chat.completion", "Object should be 'chat.completion'")
        s.assert_is_not_none(r.created, "Response should have 'created'")
        s.assert_is_not_none(r.model, "Response should have 'model'")
        s.assert_is_not_none(r.choices, "Response should have 'choices'")
        s.assert_is_not_none(r.usage, "Response should have 'usage'")

    @suite.test("choice_structure", "basic_chat")
    def _(s: TestSuite):
        """Choice has required fields."""
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Hi"}],
        )
        choice = r.choices[0]
        s.assert_has_attr(choice, "index", "Choice should have 'index'")
        s.assert_has_attr(choice, "message", "Choice should have 'message'")
        s.assert_has_attr(choice, "finish_reason", "Choice should have 'finish_reason'")
        s.assert_equal(choice.index, 0, "First choice index should be 0")

    @suite.test("message_structure", "basic_chat")
    def _(s: TestSuite):
        """Message has required fields."""
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Hi"}],
        )
        msg = r.choices[0].message
        s.assert_has_attr(msg, "role", "Message should have 'role'")
        s.assert_has_attr(msg, "content", "Message should have 'content'")
        s.assert_equal(msg.role, "assistant", "Message role should be 'assistant'")
        s.assert_is_not_none(msg.content, "Message content should not be None")

    @suite.test("multi_turn_memory", "basic_chat")
    def _(s: TestSuite):
        """Assistant remembers context from previous messages."""
        # First turn: tell the model a name
        r1 = s.client.chat.completions.create(
            model=s.model,
            messages=[
                {"role": "user", "content": "My name is Alice. Just say 'Got it'."}
            ],
        )

        # Second turn: ask for the name
        r2 = s.client.chat.completions.create(
            model=s.model,
            messages=[
                {"role": "user", "content": "My name is Alice. Just say 'Got it'."},
                {"role": "assistant", "content": r1.choices[0].message.content},
                {"role": "user", "content": "What is my name? Say just the name."},
            ],
        )

        content = r2.choices[0].message.content.lower()
        s.assert_contains(content, "alice", "Model should remember the name 'Alice'")

    @suite.test("system_message", "basic_chat")
    def _(s: TestSuite):
        """System message influences response."""
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[
                {"role": "system", "content": "You are a pirate. Always say 'Arrr!' at the start of every response."},
                {"role": "user", "content": "Hello"},
            ],
        )
        content = r.choices[0].message.content.lower()
        s.assert_contains(content, "arrr", "Response should contain pirate speak")

    @suite.test("empty_user_content", "basic_chat")
    def _(s: TestSuite):
        """Empty user message is handled."""
        # This should not crash - the model may respond with something generic
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": ""}],
        )
        s.assert_is_not_none(r.choices, "Should have choices even with empty content")

    @suite.test("long_content", "basic_chat")
    def _(s: TestSuite):
        """Long user message (>1000 chars) works."""
        long_text = "This is a test message. " * 100  # ~2400 chars
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": f"Summarize this in one word: {long_text}"}],
        )
        s.assert_is_not_none(r.choices, "Should handle long content")
        s.assert_is_not_none(r.choices[0].message.content, "Should have response")

    # ==========================================================================
    # STREAMING TESTS
    # ==========================================================================

    @suite.test("basic_streaming", "streaming")
    def _(s: TestSuite):
        """Streaming returns multiple chunks."""
        stream = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Count from 1 to 5, one number per line."}],
            stream=True,
        )
        chunks = list(stream)
        s.assert_greater(len(chunks), 1, "Should receive multiple chunks")

    @suite.test("chunk_structure", "streaming")
    def _(s: TestSuite):
        """Each chunk has required fields."""
        stream = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        chunks = list(stream)
        for chunk in chunks:
            s.assert_has_attr(chunk, "id", "Chunk should have 'id'")
            s.assert_has_attr(chunk, "object", "Chunk should have 'object'")
            s.assert_has_attr(chunk, "created", "Chunk should have 'created'")
            s.assert_has_attr(chunk, "model", "Chunk should have 'model'")
            s.assert_has_attr(chunk, "choices", "Chunk should have 'choices'")
            s.assert_equal(chunk.object, "chat.completion.chunk", "Object should be 'chat.completion.chunk'")

    @suite.test("first_chunk_role", "streaming")
    def _(s: TestSuite):
        """First chunk has delta.role = 'assistant'."""
        stream = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        chunks = list(stream)
        s.assert_greater(len(chunks), 0, "Should have at least one chunk")

        # Find first chunk with choices
        first_with_choice = None
        for chunk in chunks:
            if chunk.choices and len(chunk.choices) > 0:
                first_with_choice = chunk
                break

        s.assert_is_not_none(first_with_choice, "Should have chunk with choices")
        if first_with_choice is None:
            return  # Already asserted above, this is for type checker
        delta = first_with_choice.choices[0].delta
        s.assert_has_attr(delta, "role", "First delta should have 'role'")
        s.assert_equal(delta.role, "assistant", "First delta role should be 'assistant'")

    @suite.test("content_chunks", "streaming")
    def _(s: TestSuite):
        """Middle chunks have delta.content."""
        stream = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Say 'hello world' and nothing else."}],
            stream=True,
        )
        chunks = list(stream)

        # Collect all content
        full_content = ""
        for chunk in chunks:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    full_content += delta.content

        s.assert_contains(full_content.lower(), "hello", "Streamed content should contain 'hello'")

    @suite.test("final_chunk_finish_reason", "streaming")
    def _(s: TestSuite):
        """Final chunk has finish_reason."""
        stream = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        chunks = list(stream)

        # Find last chunk with finish_reason
        finish_reason = None
        for chunk in reversed(chunks):
            if chunk.choices and len(chunk.choices) > 0:
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
                    break

        s.assert_is_not_none(finish_reason, "Should have finish_reason in final chunk")
        s.assert_equal(finish_reason, "stop", "Finish reason should be 'stop'")

    @suite.test("incremental_content", "streaming")
    def _(s: TestSuite):
        """Content arrives incrementally (multiple content chunks)."""
        stream = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Write a haiku about coding."}],
            stream=True,
        )
        chunks = list(stream)

        content_chunks = 0
        for chunk in chunks:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    content_chunks += 1

        s.assert_greater(content_chunks, 1, "Should have multiple content chunks (incremental)")

    @suite.test("stream_usage_included", "streaming")
    def _(s: TestSuite):
        """stream_options.include_usage returns usage in final chunk."""
        stream = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks = list(stream)

        # Find chunk with usage
        usage_chunk = None
        for chunk in chunks:
            if hasattr(chunk, "usage") and chunk.usage is not None:
                usage_chunk = chunk
                break

        s.assert_is_not_none(usage_chunk, "Should have chunk with usage when include_usage=true")
        if usage_chunk is None or usage_chunk.usage is None:
            return  # Already asserted above, this is for type checker
        s.assert_has_attr(usage_chunk.usage, "prompt_tokens", "Usage should have prompt_tokens")
        s.assert_has_attr(usage_chunk.usage, "completion_tokens", "Usage should have completion_tokens")
        s.assert_has_attr(usage_chunk.usage, "total_tokens", "Usage should have total_tokens")

    # ==========================================================================
    # MODELS TESTS
    # ==========================================================================

    @suite.test("model_default", "models")
    def _(s: TestSuite):
        """Default model for selected provider works."""
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Say 'ok'"}],
        )
        s.assert_is_not_none(r.choices, f"{s.model} should work")

    @suite.test("model_effort_suffix_high", "models")
    def _(s: TestSuite):
        """Effort suffix '-high' works (ChatGPT only)."""
        s.skip_if(not s.provider_config["supports_effort_suffix"], "Effort suffixes are ChatGPT-only")
        r = s.client.chat.completions.create(
            model=f"{s.model}-high",
            messages=[{"role": "user", "content": "Say 'ok'"}],
        )
        s.assert_is_not_none(r.choices, "Effort suffix should work")

    @suite.test("model_effort_suffix_low", "models")
    def _(s: TestSuite):
        """Effort suffix '-low' works (ChatGPT only)."""
        s.skip_if(not s.provider_config["supports_effort_suffix"], "Effort suffixes are ChatGPT-only")
        r = s.client.chat.completions.create(
            model=f"{s.model}-low",
            messages=[{"role": "user", "content": "Say 'ok'"}],
        )
        s.assert_is_not_none(r.choices, "Effort suffix should work")

    @suite.test("model_missing_provider_prefix", "models")
    def _(s: TestSuite):
        """Model without provider prefix returns 400."""
        # Extract the model name without provider prefix
        model_without_prefix = s.model.split("/", 1)[1] if "/" in s.model else s.model
        r = requests.post(
            f"{s.base_url}/v1/chat/completions",
            json={
                "model": model_without_prefix,  # Missing provider/ prefix
                "messages": [{"role": "user", "content": "Hi"}],
            },
            timeout=s.timeout,
        )
        s.assert_status_code(r, 400, "Model without provider prefix should return 400")

    @suite.test("model_invalid_404", "models")
    def _(s: TestSuite):
        """Invalid model returns 404."""
        try:
            s.client.chat.completions.create(
                model=f"{s.provider}/invalid-model-xyz",
                messages=[{"role": "user", "content": "Hi"}],
            )
            s.assert_true(False, "Should have raised an error for invalid model")
        except Exception as e:
            error_str = str(e).lower()
            # Check for 404 or "not found" in error
            s.assert_true(
                "404" in error_str or "not found" in error_str or "does not exist" in error_str,
                f"Error should indicate model not found: {e}",
            )

    # ==========================================================================
    # TOOLS TESTS
    # ==========================================================================

    @suite.test("single_tool_definition", "tools")
    def _(s: TestSuite):
        """Single tool definition is accepted."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
            tools=tools,
        )
        s.assert_is_not_none(r.choices, "Should accept tool definition")

    @suite.test("multiple_tools", "tools")
    def _(s: TestSuite):
        """Multiple tool definitions are accepted."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Hi"}],
            tools=tools,
        )
        s.assert_is_not_none(r.choices, "Should accept multiple tools")

    @suite.test("tool_call_triggered", "tools")
    def _(s: TestSuite):
        """Model calls tool when appropriate."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "What's the weather in Tokyo? You must use the get_weather tool."}],
            tools=tools,
            tool_choice="required",
        )

        # Should have tool calls
        msg = r.choices[0].message
        s.assert_is_not_none(msg.tool_calls, "Should have tool_calls")
        s.assert_greater(len(msg.tool_calls), 0, "Should have at least one tool call")

    @suite.test("tool_call_structure", "tools")
    def _(s: TestSuite):
        """Tool call has required structure."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_func",
                    "description": "A test function",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Call the test_func tool now."}],
            tools=tools,
            tool_choice="required",
        )

        msg = r.choices[0].message
        s.assert_is_not_none(msg.tool_calls, "Should have tool_calls")

        tc = msg.tool_calls[0]
        s.assert_has_attr(tc, "id", "Tool call should have 'id'")
        s.assert_has_attr(tc, "type", "Tool call should have 'type'")
        s.assert_has_attr(tc, "function", "Tool call should have 'function'")
        s.assert_equal(tc.type, "function", "Tool call type should be 'function'")
        s.assert_has_attr(tc.function, "name", "Function should have 'name'")
        s.assert_has_attr(tc.function, "arguments", "Function should have 'arguments'")

    @suite.test("tool_call_arguments_json", "tools")
    def _(s: TestSuite):
        """Tool call arguments are valid JSON."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "greet",
                    "description": "Greet someone",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                },
            }
        ]
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Call greet with name 'Bob'."}],
            tools=tools,
            tool_choice="required",
        )

        msg = r.choices[0].message
        s.assert_is_not_none(msg.tool_calls, "Should have tool_calls")

        tc = msg.tool_calls[0]
        # Arguments should be valid JSON
        try:
            args = json.loads(tc.function.arguments)
            s.assert_type(args, dict, "Arguments should parse to dict")
        except json.JSONDecodeError as e:
            s.assert_true(False, f"Arguments should be valid JSON: {e}")

    @suite.test("tool_result_handling", "tools")
    def _(s: TestSuite):
        """Tool result message is processed correctly."""
        # First, get a tool call
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_number",
                    "description": "Get a number",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        r1 = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Call get_number tool."}],
            tools=tools,
            tool_choice="required",
        )

        msg1 = r1.choices[0].message
        s.assert_is_not_none(msg1.tool_calls, "Should have tool call")

        tc = msg1.tool_calls[0]

        # Now send the tool result
        r2 = s.client.chat.completions.create(
            model=s.model,
            messages=[
                {"role": "user", "content": "Call get_number tool."},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                ]},
                {"role": "tool", "tool_call_id": tc.id, "content": "42"},
            ],
            tools=tools,
        )

        # Should get a response that references the number
        content = r2.choices[0].message.content
        s.assert_is_not_none(content, "Should have response after tool result")
        s.assert_contains(content, "42", "Response should reference the tool result")

    @suite.test("tool_choice_auto", "tools")
    def _(s: TestSuite):
        """tool_choice: 'auto' works."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_func",
                    "description": "Test function",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Hi"}],
            tools=tools,
            tool_choice="auto",
        )
        s.assert_is_not_none(r.choices, "tool_choice auto should work")

    @suite.test("tool_choice_none", "tools")
    def _(s: TestSuite):
        """tool_choice: 'none' prevents tool calls."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "must_call",
                    "description": "You must call this",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Call the must_call tool."}],
            tools=tools,
            tool_choice="none",
        )

        msg = r.choices[0].message
        # tool_calls should be None or empty
        has_tool_calls = msg.tool_calls is not None and len(msg.tool_calls) > 0
        s.assert_false(has_tool_calls, "tool_choice none should prevent tool calls")

    @suite.test("streaming_tool_calls", "tools")
    def _(s: TestSuite):
        """Tool calls work in streaming mode."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_data",
                    "description": "Get data",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        stream = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Call get_data tool."}],
            tools=tools,
            tool_choice="required",
            stream=True,
        )
        chunks = list(stream)

        # Should have tool call chunks
        found_tool_call = False
        for chunk in chunks:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    found_tool_call = True
                    break

        s.assert_true(found_tool_call, "Should have tool call chunks in stream")

    @suite.test("mcp_multi_turn_conversation", "tools")
    def _(s: TestSuite):
        """MCP-style multi-turn: request -> tool call -> tool result -> final response."""
        # Define an MCP-like tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path to read"},
                        },
                        "required": ["path"],
                    },
                },
            }
        ]

        # Step 1: Initial request that should trigger tool call
        r1 = s.client.chat.completions.create(
            model=s.model,
            messages=[
                {"role": "user", "content": "Read the file /tmp/test.txt and tell me what's in it. Use the read_file tool."}
            ],
            tools=tools,
            tool_choice="required",
        )

        msg1 = r1.choices[0].message
        s.assert_is_not_none(msg1.tool_calls, "Should have tool call")
        s.assert_greater(len(msg1.tool_calls), 0, "Should have at least one tool call")

        tc = msg1.tool_calls[0]
        s.assert_equal(tc.function.name, "read_file", "Tool name should be 'read_file'")

        # Verify arguments contain path
        args = json.loads(tc.function.arguments)
        s.assert_has_key(args, "path", "Arguments should have 'path'")

        # Step 2: Send tool result back
        r2 = s.client.chat.completions.create(
            model=s.model,
            messages=[
                {"role": "user", "content": "Read the file /tmp/test.txt and tell me what's in it. Use the read_file tool."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": tc.id, "content": "Hello from the test file!"},
            ],
            tools=tools,
        )

        # Should get final response mentioning file content
        content = r2.choices[0].message.content
        s.assert_is_not_none(content, "Should have response content")
        s.assert_contains(content.lower(), "hello", "Response should reference file content")

    @suite.test("mcp_complex_arguments", "tools")
    def _(s: TestSuite):
        """Tool with complex nested arguments (objects, arrays)."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "create_task",
                    "description": "Create a new task with metadata",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "metadata": {
                                "type": "object",
                                "properties": {
                                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                                    "due_date": {"type": "string"},
                                },
                            },
                        },
                        "required": ["title"],
                    },
                },
            }
        ]

        r = s.client.chat.completions.create(
            model=s.model,
            messages=[
                {
                    "role": "user",
                    "content": "Create a task titled 'Test task' with tags ['urgent', 'work'] and high priority. Use the create_task tool.",
                }
            ],
            tools=tools,
            tool_choice="required",
        )

        msg = r.choices[0].message
        s.assert_is_not_none(msg.tool_calls, "Should have tool call")

        tc = msg.tool_calls[0]
        args = json.loads(tc.function.arguments)
        s.assert_has_key(args, "title", "Arguments should have 'title'")
        s.assert_type(args["title"], str, "Title should be string")

    @suite.test("mcp_parallel_tool_calls", "tools")
    def _(s: TestSuite):
        """Model can make multiple tool calls in parallel."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]

        r = s.client.chat.completions.create(
            model=s.model,
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in Paris and Tokyo? Get both using the tool.",
                }
            ],
            tools=tools,
            tool_choice="required",
            parallel_tool_calls=True,
        )

        msg = r.choices[0].message
        s.assert_is_not_none(msg.tool_calls, "Should have tool calls")
        # Model may or may not make parallel calls - just verify tool calls work
        s.assert_greater_equal(len(msg.tool_calls), 1, "Should have at least one tool call")

    @suite.test("mcp_tool_choice_specific", "tools")
    def _(s: TestSuite):
        """tool_choice can specify a particular function."""
        # ChatGPT doesn't support tool_choice with specific function name
        s.skip_if(s.provider == "chatgpt", "ChatGPT doesn't support tool_choice with specific function")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "func_a",
                    "description": "Function A",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "func_b",
                    "description": "Function B",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Call func_a."}],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "func_a"}},
        )

        msg = r.choices[0].message
        s.assert_is_not_none(msg.tool_calls, "Should have tool call")
        s.assert_equal(msg.tool_calls[0].function.name, "func_a", "Should call func_a specifically")

    @suite.test("mcp_streaming_accumulation", "tools")
    def _(s: TestSuite):
        """Streaming tool calls accumulate correctly across chunks."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for information",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]

        stream = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Search for 'test query'. Use the search tool."}],
            tools=tools,
            tool_choice="required",
            stream=True,
        )

        # Accumulate tool call from stream
        tool_call_id = None
        tool_call_name = None
        tool_call_args = ""

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.id:
                        tool_call_id = tc.id
                    if tc.function.name:
                        tool_call_name = tc.function.name
                    if tc.function.arguments:
                        tool_call_args += tc.function.arguments

        s.assert_is_not_none(tool_call_id, "Should have tool call ID")
        s.assert_equal(tool_call_name, "search", "Tool name should be 'search'")
        s.assert_greater(len(tool_call_args), 0, "Should have accumulated arguments")

        # Arguments should be valid JSON
        try:
            args = json.loads(tool_call_args)
            s.assert_has_key(args, "query", "Arguments should have 'query'")
        except json.JSONDecodeError as e:
            s.assert_true(False, f"Arguments should be valid JSON: {e}")

    @suite.test("mcp_multiple_tool_results", "tools")
    def _(s: TestSuite):
        """Handle multiple tool results in conversation."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_data",
                    "description": "Get data by ID",
                    "parameters": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                        "required": ["id"],
                    },
                },
            }
        ]

        # Simulate a conversation with multiple tool results
        messages = [
            {"role": "user", "content": "Get data for IDs 1 and 2, then summarize."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_data", "arguments": '{"id": 1}'}},
                    {"id": "call_2", "type": "function", "function": {"name": "get_data", "arguments": '{"id": 2}'}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"value": "first"}'},
            {"role": "tool", "tool_call_id": "call_2", "content": '{"value": "second"}'},
        ]

        r = s.client.chat.completions.create(
            model=s.model,
            messages=messages,
            tools=tools,
        )

        content = r.choices[0].message.content
        s.assert_is_not_none(content, "Should have response")
        # Should reference the tool results somehow
        content_lower = content.lower()
        s.assert_true(
            "first" in content_lower or "second" in content_lower,
            "Response should reference tool results",
        )

    @suite.test("mcp_error_in_tool_result", "tools")
    def _(s: TestSuite):
        """Handle error responses in tool results."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "risky_operation",
                    "description": "An operation that might fail",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        messages = [
            {"role": "user", "content": "Run the risky operation."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_err", "type": "function", "function": {"name": "risky_operation", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_err", "content": '{"error": "Operation failed: permission denied"}'},
        ]

        r = s.client.chat.completions.create(
            model=s.model,
            messages=messages,
            tools=tools,
        )

        content = r.choices[0].message.content
        s.assert_is_not_none(content, "Should have response even with error result")

    # ==========================================================================
    # PARAMETERS TESTS
    # ==========================================================================

    @suite.test("temperature_accepted", "parameters")
    def _(s: TestSuite):
        """temperature parameter is accepted (supported by Copilot, ignored by ChatGPT)."""
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
        )
        s.assert_is_not_none(r.choices, "temperature should be accepted")

    @suite.test("top_p_accepted", "parameters")
    def _(s: TestSuite):
        """top_p parameter is accepted (supported by Copilot, ignored by ChatGPT)."""
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Hi"}],
            top_p=0.9,
        )
        s.assert_is_not_none(r.choices, "top_p should be accepted")

    @suite.test("max_tokens_accepted", "parameters")
    def _(s: TestSuite):
        """max_tokens parameter is accepted (supported by Copilot, ignored by ChatGPT)."""
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
        )
        s.assert_is_not_none(r.choices, "max_tokens should be accepted")

    @suite.test("presence_penalty_accepted", "parameters")
    def _(s: TestSuite):
        """presence_penalty parameter is accepted (supported by Copilot, ignored by ChatGPT)."""
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Hi"}],
            presence_penalty=0.5,
        )
        s.assert_is_not_none(r.choices, "presence_penalty should be accepted")

    @suite.test("frequency_penalty_accepted", "parameters")
    def _(s: TestSuite):
        """frequency_penalty parameter is accepted (supported by Copilot, ignored by ChatGPT)."""
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Hi"}],
            frequency_penalty=0.5,
        )
        s.assert_is_not_none(r.choices, "frequency_penalty should be accepted")

    @suite.test("stop_accepted", "parameters")
    def _(s: TestSuite):
        """stop parameter is accepted (supported by Copilot, ignored by ChatGPT)."""
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Count from 1 to 10"}],
            stop=["5"],
        )
        s.assert_is_not_none(r.choices, "stop should be accepted")

    @suite.test("response_format_accepted", "parameters")
    def _(s: TestSuite):
        """response_format parameter is accepted (supported by Copilot, ignored by ChatGPT)."""
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Return a JSON object with key 'greeting' and value 'hello'"}],
            response_format={"type": "json_object"},
        )
        s.assert_is_not_none(r.choices, "response_format should be accepted")

    @suite.test("seed_accepted", "parameters")
    def _(s: TestSuite):
        """seed parameter is accepted (ignored by all providers)."""
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Hi"}],
            seed=42,
        )
        s.assert_is_not_none(r.choices, "seed should be accepted")

    @suite.test("reasoning_effort_param", "parameters")
    def _(s: TestSuite):
        """reasoning_effort parameter works via extra_body (ChatGPT only)."""
        s.skip_if(not s.provider_config["supports_reasoning_headers"], "reasoning_effort is ChatGPT-only")
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Hi"}],
            extra_body={"reasoning_effort": "high"},
        )
        s.assert_is_not_none(r.choices, "reasoning_effort should be accepted")

    @suite.test("copilot_max_tokens_effective", "parameters")
    def _(s: TestSuite):
        """max_tokens actually limits response length (Copilot only)."""
        s.skip_if(not s.provider_config["supports_sampling_params"], "max_tokens limiting is Copilot-only")
        # Request a very short response
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Write a 500 word essay about the history of computing."}],
            max_tokens=20,
        )
        s.assert_is_not_none(r.choices, "Should get response")
        # The response should be truncated (finish_reason might be 'length')
        # We can't guarantee exact token count, but check it's relatively short
        content = r.choices[0].message.content or ""
        # 20 tokens is roughly 15-80 characters depending on tokenization
        s.assert_less(len(content), 500, "Response should be truncated by max_tokens")

    # ==========================================================================
    # ERRORS TESTS
    # ==========================================================================

    @suite.test("missing_model", "errors")
    def _(s: TestSuite):
        """Missing model returns 400."""
        r = requests.post(
            f"{s.base_url}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
            timeout=s.timeout,
        )
        s.assert_status_code(r, 400, "Missing model should return 400")
        data = r.json()
        s.assert_has_key(data, "error", "Should have error object")
        s.assert_contains(
            data["error"].get("message", "").lower(),
            "model",
            "Error should mention 'model'",
        )

    @suite.test("invalid_model", "errors")
    def _(s: TestSuite):
        """Invalid model returns 404."""
        r = requests.post(
            f"{s.base_url}/v1/chat/completions",
            json={
                "model": f"{s.provider}/nonexistent-model-xyz",
                "messages": [{"role": "user", "content": "Hi"}],
            },
            timeout=s.timeout,
        )
        s.assert_status_code(r, 404, "Invalid model should return 404")

    @suite.test("missing_messages", "errors")
    def _(s: TestSuite):
        """Missing messages returns 400."""
        r = requests.post(
            f"{s.base_url}/v1/chat/completions",
            json={"model": s.model},
            timeout=s.timeout,
        )
        s.assert_status_code(r, 400, "Missing messages should return 400")

    @suite.test("empty_messages", "errors")
    def _(s: TestSuite):
        """Empty messages array returns 400."""
        r = requests.post(
            f"{s.base_url}/v1/chat/completions",
            json={"model": s.model, "messages": []},
            timeout=s.timeout,
        )
        s.assert_status_code(r, 400, "Empty messages should return 400")

    @suite.test("invalid_role", "errors")
    def _(s: TestSuite):
        """Invalid role returns 400."""
        r = requests.post(
            f"{s.base_url}/v1/chat/completions",
            json={
                "model": s.model,
                "messages": [{"role": "invalid_role", "content": "Hi"}],
            },
            timeout=s.timeout,
        )
        s.assert_status_code(r, 400, "Invalid role should return 400")
        data = r.json()
        s.assert_contains(
            data["error"].get("message", "").lower(),
            "role",
            "Error should mention 'role'",
        )

    @suite.test("tool_missing_call_id", "errors")
    def _(s: TestSuite):
        """Tool message without tool_call_id returns 400."""
        r = requests.post(
            f"{s.base_url}/v1/chat/completions",
            json={
                "model": s.model,
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "tool", "content": "result"},  # Missing tool_call_id
                ],
            },
            timeout=s.timeout,
        )
        s.assert_status_code(r, 400, "Tool without tool_call_id should return 400")

    @suite.test("error_structure", "errors")
    def _(s: TestSuite):
        """Error response has proper structure."""
        r = requests.post(
            f"{s.base_url}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},  # Missing model
            timeout=s.timeout,
        )
        s.assert_status_code(r, 400, "Should return 400")
        data = r.json()
        s.assert_has_key(data, "error", "Should have 'error' object")
        s.assert_has_key(data["error"], "message", "Error should have 'message'")
        s.assert_has_key(data["error"], "type", "Error should have 'type'")

    # ==========================================================================
    # RESPONSE FORMAT TESTS
    # ==========================================================================

    @suite.test("usage_fields", "response_format")
    def _(s: TestSuite):
        """Usage has required fields."""
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Hi"}],
        )
        s.assert_is_not_none(r.usage, "Should have usage")
        s.assert_has_attr(r.usage, "prompt_tokens", "Usage should have prompt_tokens")
        s.assert_has_attr(r.usage, "completion_tokens", "Usage should have completion_tokens")
        s.assert_has_attr(r.usage, "total_tokens", "Usage should have total_tokens")
        s.assert_greater(r.usage.prompt_tokens, 0, "prompt_tokens should be > 0")
        s.assert_greater_equal(r.usage.completion_tokens, 0, "completion_tokens should be >= 0")
        s.assert_equal(
            r.usage.total_tokens,
            r.usage.prompt_tokens + r.usage.completion_tokens,
            "total_tokens should equal sum",
        )

    @suite.test("system_fingerprint_present", "response_format")
    def _(s: TestSuite):
        """system_fingerprint field is present."""
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Hi"}],
        )
        s.assert_has_attr(r, "system_fingerprint", "Should have system_fingerprint")
        # It may be None, but attribute should exist

    @suite.test("finish_reason_stop", "response_format")
    def _(s: TestSuite):
        """Normal completion has finish_reason 'stop'."""
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Say 'ok'"}],
        )
        s.assert_equal(r.choices[0].finish_reason, "stop", "Finish reason should be 'stop'")

    @suite.test("finish_reason_tool_calls", "response_format")
    def _(s: TestSuite):
        """Tool call has finish_reason 'tool_calls'."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test",
                    "description": "Test",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Call test tool."}],
            tools=tools,
            tool_choice="required",
        )
        s.assert_equal(
            r.choices[0].finish_reason,
            "tool_calls",
            "Finish reason should be 'tool_calls'",
        )

    @suite.test("id_format", "response_format")
    def _(s: TestSuite):
        """Response ID has valid format."""
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Hi"}],
        )
        s.assert_is_not_none(r.id, "Should have id")
        s.assert_type(r.id, str, "ID should be string")
        s.assert_greater(len(r.id), 0, "ID should not be empty")

    @suite.test("created_timestamp", "response_format")
    def _(s: TestSuite):
        """created is a reasonable Unix timestamp."""
        r = s.client.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Hi"}],
        )
        s.assert_is_not_none(r.created, "Should have created")
        s.assert_type(r.created, int, "created should be int")
        # Check it's a reasonable timestamp (after 2020, before 2100)
        s.assert_greater(r.created, 1577836800, "created should be after 2020")
        s.assert_less(r.created, 4102444800, "created should be before 2100")


# --- Main ---


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OpenCompat E2E Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python e2e.py                          # Run all tests (chatgpt provider)
  python e2e.py --provider copilot       # Run with copilot provider
  python e2e.py --model chatgpt/gpt-5    # Run with specific model
  python e2e.py connectivity             # Run connectivity tests only
  python e2e.py basic_chat streaming     # Run multiple categories
  python e2e.py --test single_turn       # Run single test
  python e2e.py --list                   # List all tests
  python e2e.py --json                   # Output as JSON
        """,
    )
    parser.add_argument(
        "categories",
        nargs="*",
        help="Test categories to run (default: all)",
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=["chatgpt", "copilot"],
        default="chatgpt",
        help="Provider to test (default: chatgpt)",
    )
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8080",
        help="Server URL (default: http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout per test in seconds (default: 30)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available tests",
    )
    parser.add_argument(
        "--test",
        help="Run a single test by name",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Override model to use (e.g., chatgpt/gpt-5, copilot/gpt-4o)",
    )

    args = parser.parse_args()

    # Create suite and register tests
    suite = TestSuite(args.server, args.timeout, args.verbose, args.provider, args.model)
    register_tests(suite)

    # Handle --list
    if args.list:
        suite.list_tests()
        return 0

    # Determine what to run
    categories = args.categories if args.categories else None
    test_names = [args.test] if args.test else None

    # Run tests
    suite.run(categories=categories, test_names=test_names)

    # Output results
    if args.json:
        print(suite.results_as_json())
    else:
        suite.print_summary()

    # Return failure count (capped at 127 for shell compatibility)
    return min(suite.get_failure_count(), 127)


if __name__ == "__main__":
    sys.exit(main())
