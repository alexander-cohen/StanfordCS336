import contextlib
import datetime as dt
import json
import os
import sqlite3
import time
from pathlib import Path

ACTIVE = {"submitting", "queued", "running"}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(exist_ok=True)
        with self.transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, record TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            db.execute(
                "INSERT OR IGNORE INTO settings VALUES ('budget_seconds', '43200')"
            )

    @contextlib.contextmanager
    def transaction(self):
        # Rollback journal (not WAL): readers/writers can live on different Torch nodes.
        db = sqlite3.connect(self.root / "ledger.sqlite3", timeout=60)
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def records(self, db=None):
        if db is not None:
            return [
                json.loads(row[0])
                for row in db.execute("SELECT record FROM runs ORDER BY id")
            ]
        with self.transaction() as conn:
            return self.records(conn)

    def get(self, experiment_id, db=None):
        records = self.records(db)
        for record in records:
            if record["experiment_id"] == experiment_id:
                return record
        raise ValueError(f"unknown experiment: {experiment_id}")

    def budget(self, db=None):
        if db is None:
            with self.transaction() as conn:
                return self.budget(conn)
        records = self.records(db)
        total = float(
            db.execute(
                "SELECT value FROM settings WHERE key='budget_seconds'"
            ).fetchone()[0]
        )
        experiments = [r for r in records if r["kind"] == "experiment"]
        spent = sum(
            r["charged_seconds"]
            for r in experiments
            if r["status"]["status_type"] not in ACTIVE
        )
        reserved = sum(
            r["training_config"]["max_runtime_seconds"]
            for r in experiments
            if r["status"]["status_type"] in ACTIVE
        )

        def job_seconds(r):
            if r["status"]["status_type"] in ACTIVE and r.get("job_started_unix"):
                return max(
                    r.get("job_elapsed_seconds", 0), time.time() - r["job_started_unix"]
                )
            return r.get("job_elapsed_seconds", 0)

        def train_seconds(r):
            if r["status"]["status_type"] in ACTIVE and r.get("training_started_unix"):
                return max(0, time.time() - r["training_started_unix"])
            return r["status"].get("used_runtime_seconds", 0)

        return {
            "used_seconds": spent + reserved,
            "spent_seconds": spent,
            "reserved_seconds": reserved,
            "remaining_seconds": total - spent - reserved,
            "total_budget_seconds": total,
            "total_training_seconds": sum(train_seconds(r) for r in experiments),
            "total_job_seconds": sum(job_seconds(r) for r in experiments),
            "final_job_seconds": sum(
                job_seconds(r) for r in records if r["kind"] == "final"
            ),
            "updated_at": now(),
        }

    def publish(self, db, record=None):
        if record is not None:
            run = self.run_dir(record["experiment_id"])
            run.mkdir(parents=True, exist_ok=True)
            atomic_json(run / "result.json", record)
        atomic_json(self.root / "budget.json", self.budget(db))

    def reserve(self, config, config_id, kind, predicted_loss=None):
        with self.transaction() as db:
            records = self.records(db)
            if kind == "experiment" and any(
                r["kind"] == kind
                and r["config_id"] == config_id
                and r["status"]["status_type"] != "submission_failed"
                and not r.get("retry_allowed", False)
                for r in records
            ):
                raise ValueError("this configuration has already been submitted")
            if kind == "experiment":
                if config["max_runtime_seconds"] > min(
                    43200, self.budget(db)["remaining_seconds"]
                ):
                    raise ValueError(
                        "requested runtime exceeds the remaining experiment budget or 12-hour limit"
                    )
            elif config["max_runtime_seconds"] > 172800:
                raise ValueError("final training runtime must be at most 48 hours")
            if (
                kind == "final"
                and sum(
                    r["kind"] == "final" and r["status"]["status_type"] in ACTIVE
                    for r in records
                )
                >= 2
            ):
                raise ValueError("at most two final jobs may be active")
            eid = max((r["experiment_id"] for r in records), default=0) + 1
            record = {
                "experiment_id": eid,
                "config_id": config_id,
                "kind": kind,
                "training_config": config,
                "predicted_final_loss": predicted_loss,
                "status": {
                    "status_type": "submitting",
                    "queued_at": now(),
                    "val_losses": [],
                },
                "charged_seconds": 0.0,
                "job_elapsed_seconds": 0.0,
            }
            db.execute("INSERT INTO runs VALUES (?, ?)", (eid, json.dumps(record)))
            self.publish(db, record)
            return record

    def update(self, experiment_id, mutate):
        with self.transaction() as db:
            record = self.get(experiment_id, db)
            mutate(record)
            db.execute(
                "UPDATE runs SET record=? WHERE id=?",
                (json.dumps(record, allow_nan=False), experiment_id),
            )
            self.publish(db, record)
            return record

    def finish(
        self, experiment_id, status_type, elapsed, reason=None, *, metrics=None, **extra
    ):
        def mutate(record):
            maximum = record["training_config"]["max_runtime_seconds"]
            record["charged_seconds"] = min(maximum, max(1.0, elapsed))
            if metrics is not None:
                record["metrics"] = metrics
            record["status"].update(
                status_type=status_type,
                used_runtime_seconds=elapsed,
                finished_at=now(),
                **extra,
            )
            if reason:
                record["status"]["reason"] = reason

        return self.update(experiment_id, mutate)

    def run_dir(self, experiment_id):
        return self.root / "runs" / str(int(experiment_id))
