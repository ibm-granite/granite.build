"""In-memory mock LineageService for testing WandBLineageStore event-building
logic without hitting the real wandb API."""

from typing import Dict, List, Optional, Tuple

from gbserver.lineage.openlineage_service import LineageService


class MockLineageService(LineageService):
    """Records emitted events in memory and answers count/search queries locally."""

    def __init__(self):
        self._events: List[Dict] = []

    def emit_event(self, event: Dict) -> None:
        self._events.append(event)

    def _tags_for_event(self, event: Dict) -> set:
        tags_dict = event.get("run", {}).get("facets", {}).get("tags", {})
        return {f"{k}={v}" for k, v in tags_dict.items() if not k.startswith("_")}

    def count_runs_by_tags(
        self, tags: List[str], required_tags: Optional[List[str]] = None
    ) -> int:
        tag_set = set(tags)
        required = set(required_tags or [])
        seen_runs = set()
        for event in self._events:
            run_id = event.get("run", {}).get("runId", "")
            if run_id in seen_runs:
                continue
            event_tags = self._tags_for_event(event)
            if tag_set & event_tags and required.issubset(event_tags):
                seen_runs.add(run_id)
        return len(seen_runs)

    def count_events_by_tags(
        self, tags: List[str], required_tags: Optional[List[str]] = None
    ) -> int:
        tag_set = set(tags)
        required = set(required_tags or [])
        total = 0
        for event in self._events:
            event_tags = self._tags_for_event(event)
            if tag_set & event_tags and required.issubset(event_tags):
                total += 1
        return total

    def filter_unrecorded(
        self,
        target_ids: set,
        expected_counts: Optional[Dict[str, int]] = None,
        on_query_error=None,
    ) -> set:
        # Mirror the real service: count the distinct runs carrying each
        # ``target_id=<uuid>`` tag (emitted events are the in-memory stand-in for
        # wandb run metadata), then treat a candidate as recorded only when its
        # run count meets its expected count. A ``None`` expected count (or a
        # missing key) falls back to the presence check (recorded once >=1 run).
        run_counts: Dict[str, int] = {}
        seen: set = set()  # (target_id, run_id) so each run counts once per target
        for event in self._events:
            run_id = event.get("run", {}).get("runId", "")
            for tag in self._tags_for_event(event):
                if not tag.startswith("target_id="):
                    continue
                tid = tag.split("=", 1)[1]
                if (tid, run_id) in seen:
                    continue
                seen.add((tid, run_id))
                run_counts[tid] = run_counts.get(tid, 0) + 1
        recorded: set = set()
        for tid in target_ids:
            count = run_counts.get(tid, 0)
            if count == 0:
                continue
            expected = (expected_counts or {}).get(tid)
            if expected is None or count >= expected:
                recorded.add(tid)
        return set(target_ids) - recorded

    def search_lineage_by_tags(
        self, tags: List[str], limit: int = 10, offset: int = 0
    ) -> Tuple[int, List[Dict]]:
        tag_set = set(tags)
        matching = [e for e in self._events if tag_set & self._tags_for_event(e)]
        return len(matching), matching[offset : offset + limit]

    def get_artifact_graph(
        self,
        artifact_name: Optional[str] = None,
        artifact_url: Optional[str] = None,
        artifact_type: Optional[str] = None,
        max_depth: int = 10,
        direction: str = "downstream",
    ) -> Optional[Dict]:
        return None
