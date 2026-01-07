#!/usr/bin/env python3
"""Morning Results Checker - Full Shakespeare Benchmark"""
import sys
import io
import json
from pathlib import Path
from datetime import datetime, timedelta

# Force UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def check_results():
    print("=" * 70)
    print("MORNING CHECK: Full Shakespeare Benchmark Results")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Find latest results
    results_dir = Path('results')
    files = sorted(results_dir.glob('shakespeare_benchmark_*.json'),
                  key=lambda x: x.stat().st_mtime, reverse=True)

    if not files:
        print("❌ No results found!")
        print("   Benchmark may still be running or failed.")
        return

    latest = files[0]
    mod_time = datetime.fromtimestamp(latest.stat().st_mtime)
    age = datetime.now() - mod_time

    print(f"📁 Latest Result: {latest.name}")
    print(f"⏰ Last Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 Age: {age}")
    print()

    # Check if it's from overnight run
    if age > timedelta(hours=12):
        print("⚠️  This result is OLD (more than 12 hours)")
        print("   Benchmark may not have started or failed.")
        print()

    # Read results
    try:
        with open(latest, 'r') as f:
            data = json.load(f)

        config = data.get('config', {})
        history = data.get('history', {})
        final = data.get('final_metrics', {})

        print("=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print()

        # Configuration
        print("📋 Configuration:")
        print(f"   Epochs: {config.get('epochs', 'N/A')}")
        print(f"   Batch Size: {config.get('batch_size', 'N/A')}")
        print(f"   Hidden Dim: {config.get('hidden_dim', 'N/A')}")
        print(f"   Embedding Dim: {config.get('embedding_dim', 'N/A')}")
        print()

        # Training Progress
        epochs_completed = len(history.get('train_loss', []))
        print(f"✅ Epochs Completed: {epochs_completed}/{config.get('epochs', '?')}")
        print()

        if epochs_completed > 0:
            # Loss metrics
            train_losses = history['train_loss']
            val_losses = history['val_loss']
            t_scores = history['t_score']
            sleep_events = history['sleep_events']

            print("📊 Training Metrics:")
            print(f"   Initial Train Loss: {train_losses[0]:.4f}")
            print(f"   Final Train Loss: {train_losses[-1]:.4f}")
            print(f"   Loss Reduction: {(train_losses[0] - train_losses[-1]):.4f}")
            print()

            print(f"   Best Val Loss: {final.get('best_val_loss', min(val_losses)):.4f}")
            print(f"   Final Val Loss: {val_losses[-1]:.4f}")
            print()

            print("🧠 T-Score (Wisdom) Metrics:")
            print(f"   Initial T-Score: {t_scores[0]:.4f}")
            print(f"   Final T-Score: {t_scores[-1]:.4f}")
            print(f"   Average T-Score: {sum(t_scores)/len(t_scores):.4f}")
            print(f"   Min T-Score: {min(t_scores):.4f}")
            print(f"   Max T-Score: {max(t_scores):.4f}")
            print()

            print("💤 Sleep Protocol:")
            print(f"   Total Sleep Events: {sum(sleep_events)}")
            print(f"   Epochs with Sleep: {sum(1 for s in sleep_events if s > 0)}")
            print(f"   Average per Epoch: {sum(sleep_events)/len(sleep_events):.1f}")
            print()

            # Training time
            training_time = final.get('training_time_minutes', 0)
            hours = int(training_time // 60)
            minutes = int(training_time % 60)
            print(f"⏱️  Training Time: {hours}h {minutes}m ({training_time:.1f} minutes)")
            print()

            # Text generation sample
            samples = data.get('samples', [])
            if samples:
                print("=" * 70)
                print("📝 TEXT GENERATION SAMPLE (Last Epoch)")
                print("=" * 70)
                print()
                last_sample = samples[-1][:500]  # First 500 chars
                print(last_sample)
                if len(samples[-1]) > 500:
                    print("...")
                print()

            # Comparison
            print("=" * 70)
            print("📈 COMPARISON TO BASELINE")
            print("=" * 70)
            print()
            print("Karpathy char-rnn (50 epochs, GPU):")
            print("   Final Loss: ~1.4")
            print()
            print(f"GodelAI ({epochs_completed} epochs, CPU):")
            print(f"   Final Loss: {train_losses[-1]:.4f}")
            print()

            if train_losses[-1] < 2.0:
                print("✅ EXCELLENT - Comparable to baseline!")
            elif train_losses[-1] < 2.5:
                print("✅ GOOD - Respectable performance on CPU")
            else:
                print("⚠️  Higher than baseline (may need more epochs)")

            print()

            # Status
            if epochs_completed == config.get('epochs'):
                print("=" * 70)
                print("🎉 BENCHMARK COMPLETED SUCCESSFULLY!")
                print("=" * 70)
            else:
                print("=" * 70)
                print(f"⚠️  INCOMPLETE - {epochs_completed}/{config.get('epochs')} epochs")
                print("=" * 70)

        else:
            print("❌ No training data found in results!")

    except Exception as e:
        print(f"❌ Error reading results: {e}")

    print()
    print("Next steps:")
    print("1. Create detailed analysis report")
    print("2. Compare mini vs full benchmark")
    print("3. Update documentation")
    print("4. Commit results to GitHub")

if __name__ == "__main__":
    check_results()
