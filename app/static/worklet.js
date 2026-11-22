// Runs on the audio thread. The AudioContext is created at 16 kHz, so the browser
// resamples the mic for us; we only batch 100 ms chunks and convert to int16.
class PCM16 extends AudioWorkletProcessor {
  constructor() {
    super();
    this.size = sampleRate / 10;
    this.buf = new Int16Array(this.size);
    this.n = 0;
  }

  process([input]) {
    const ch = input[0];
    if (!ch) return true;
    for (const x of ch) {
      this.buf[this.n++] = Math.max(-1, Math.min(1, x)) * 0x7fff;
      if (this.n === this.size) {
        this.port.postMessage(this.buf.buffer, [this.buf.buffer]);
        this.buf = new Int16Array(this.size);
        this.n = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm16", PCM16);
