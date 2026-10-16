import re
import subprocess
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "CC1080p"
EVENTS = ROOT / "logs" / EXPERIMENT
WEIGHTS = ROOT / "assets" / "weights"


def steps_per_epoch() -> int:
    checkpoints = []
    for path in WEIGHTS.glob(f"{EXPERIMENT}_e*_s*.pth"):
        match = re.search(r"_e(\d+)_s(\d+)\.pth$", path.name)
        if match:
            checkpoints.append(tuple(map(int, match.groups())))
    checkpoints.sort()
    if len(checkpoints) >= 2:
        old_epoch, old_step = checkpoints[-2]
        epoch, step = checkpoints[-1]
        if epoch > old_epoch and (step - old_step) % (epoch - old_epoch) == 0:
            return (step - old_step) // (epoch - old_epoch)
    return 145


def main() -> None:
    event_files = sorted(EVENTS.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file found in {EVENTS}")

    events = EventAccumulator(str(event_files[-1]))
    steps_per_epoch_count = steps_per_epoch()
    print(
        f"Live RVC metrics for {EXPERIMENT} (Ctrl+C stops monitoring only)",
        flush=True,
    )
    last_line = ""
    while True:
        events.Reload()
        tags = events.Tags().get("scalars", [])

        def latest(tag: str):
            values = events.Scalars(tag) if tag in tags else []
            return values[-1] if values else None

        generator_loss = latest("loss/g/total")
        if generator_loss:
            step = generator_loss.step
            epoch = (step - 1) // steps_per_epoch_count + 1
            batch = (step - 1) % steps_per_epoch_count + 1
            discriminator_loss = latest("loss/d/total")
            mel_loss = latest("loss/g/mel")
            gpu = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip().splitlines()[0].split(",")
            line = (
                f"Epoch ~{epoch}/200 | batch ~{batch}/{steps_per_epoch_count} "
                f"| step {step} | G loss {generator_loss.value:.3f} "
                f"| D loss {discriminator_loss.value:.3f} "
                f"| mel {mel_loss.value:.3f} "
                f"| GPU {gpu[0].strip()} MiB, {gpu[1].strip()}%"
            )
            if line != last_line:
                print(line, flush=True)
                last_line = line
        time.sleep(5)


if __name__ == "__main__":
    main()
