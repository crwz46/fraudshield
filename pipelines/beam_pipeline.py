"""
Apache Beam pipeline for real-time fraud feature processing.
Simulates streaming transaction processing with windowed feature aggregation.
Runs locally with DirectRunner; designed for Dataflow deployment.
"""

import json
import typing
import apache_beam as beam
from apache_beam.transforms.window import SlidingWindows
from apache_beam.transforms.trigger import AfterWatermark, AfterProcessingTime
from apache_beam.options.pipeline_options import PipelineOptions


class Transaction(typing.NamedTuple):
    transaction_id: str
    amount: float
    merchant_category: str
    merchant_country: str
    card_present: bool
    distance_from_home_km: float
    hour_of_day: int
    day_of_week: int
    timestamp: str
    card_type: str
    device_id: str
    ip_country_match: bool


class ParseTransaction(beam.DoFn):
    def process(self, line: str):
        record = json.loads(line)
        yield Transaction(
            transaction_id=record["transaction_id"],
            amount=float(record["amount"]),
            merchant_category=record["merchant_category"],
            merchant_country=record["merchant_country"],
            card_present=bool(record.get("card_present", True)),
            distance_from_home_km=float(record.get("distance_from_home_km", 0)),
            hour_of_day=int(record.get("hour_of_day", 12)),
            day_of_week=int(record.get("day_of_week", 0)),
            timestamp=record["timestamp"],
            card_type=record.get("card_type", "unknown"),
            device_id=record.get("device_id", "unknown"),
            ip_country_match=bool(record.get("ip_country_match", True)),
        )


class ComputeWindowedFeatures(beam.DoFn):
    def process(self, element, window=beam.DoFn.WindowParam):
        key, transactions = element
        txns = list(transactions)
        total_amount = sum(t.amount for t in txns)
        avg_amount = total_amount / max(len(txns), 1)
        velocity = len(txns)

        yield {
            "user_key": key,
            "window_start": window.start.to_utc_datetime().isoformat(),
            "window_end": window.end.to_utc_datetime().isoformat(),
            "transaction_count": len(txns),
            "total_amount": round(total_amount, 2),
            "avg_amount": round(avg_amount, 2),
            "max_amount": round(max(t.amount for t in txns), 2),
            "velocity": velocity,
            "unique_devices": len(set(t.device_id for t in txns)),
            "unique_countries": len(set(t.merchant_country for t in txns)),
            "card_not_present_count": sum(1 for t in txns if not t.card_present),
            "ip_mismatch_count": sum(1 for t in txns if not t.ip_country_match),
            "high_value_txns": sum(1 for t in txns if t.amount > 1000),
        }


class FormatOutput(beam.DoFn):
    def process(self, element):
        yield json.dumps(element)


def run(input_path: str, output_path: str, beam_args: list[str] = None):
    """Run the Beam pipeline locally or on Dataflow."""
    options = PipelineOptions(beam_args or [])

    with beam.Pipeline(options=options) as p:
        (
            p
            | "ReadTransactions" >> beam.io.ReadFromText(input_path)
            | "ParseJSON" >> beam.ParDo(ParseTransaction())
            | "AttachTimestamp" >> beam.Map(
                lambda t: beam.window.TimestampedValue(t, 0)
            )
            | "WindowByDevice" >> beam.WindowInto(
                SlidingWindows(size=3600, period=300),  # 1h windows, 5min sliding
                trigger=AfterWatermark(),
                allowed_lateness=120,
            )
            | "GroupByDevice" >> beam.GroupBy(lambda t: t.device_id)
            | "ComputeFeatures" >> beam.ParDo(ComputeWindowedFeatures())
            | "FormatJSON" >> beam.ParDo(FormatOutput())
            | "WriteFeatures" >> beam.io.WriteToText(
                output_path, file_name_suffix=".jsonl", shard_name_template=""
            )
        )

    print(f"Beam pipeline complete. Output: {output_path}")


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    input_path = str(Path("data/synthetic/transactions.csv"))
    output_path = str(Path("data/features/beam_output"))
    Path("data/features").mkdir(parents=True, exist_ok=True)
    run(input_path, output_path)
