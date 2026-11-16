import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { AudioPlayer } from '../../static/duplex/lib/audio-player.js';

class Context {
    constructor() {
        this.sampleRate = 24000; this.currentTime = 1; this.state = 'running';
        this.destination = {}; this.baseLatency = .01; this.outputLatency = .064;
        this.sources = [];
    }
    createBuffer(channels, count, rate) {
        const data = new Float32Array(count);
        return {duration: count/rate, getChannelData: () => data};
    }
    createBufferSource() {
        const node = {connect: vi.fn(), disconnect: vi.fn(), start: vi.fn(), stop: vi.fn()};
        this.sources.push(node); return node;
    }
    getOutputTimestamp() { return {contextTime: .9, performanceTime: 100}; }
}
const pcm = (samples) => btoa(String.fromCharCode(...new Uint8Array(new Float32Array(samples).buffer)));

describe('AudioPlayer continuity and optional observation', () => {
    let player;
    beforeEach(() => {
        vi.useFakeTimers(); vi.advanceTimersByTime(10);
        vi.stubGlobal('AudioContext', Context);
        vi.stubGlobal('requestAnimationFrame', fn => fn());
        player = new AudioPlayer({getPlaybackDelayMs: () => 0}); player.init();
    });
    afterEach(() => {player.stop(); vi.useRealTimers(); vi.unstubAllGlobals();});

    it('preserves the scheduled tail across natural turns, including PCM samples', () => {
        player.beginTurn(); player.playChunk(pcm([.1, -.2, 0, .3]));
        const first = player.ctx.sources[0], tail = player.nextTime;
        player.endTurn(); player.beginTurn(); player.playChunk(pcm([.4, 0]));
        expect(first.stop).not.toHaveBeenCalled();
        expect(player.ctx.sources[1].start).toHaveBeenCalledWith(tail);
        expect([...first.buffer.getChannelData(0)]).toEqual([...new Float32Array([.1, -.2, 0, .3])]);
    });
    it('interrupts every scheduled source and cancels a delayed first chunk', () => {
        player.beginTurn(); player.playChunk(pcm([1, 0])); player.stopAll();
        expect(player.ctx.sources[0].stop).toHaveBeenCalledOnce();
        expect(player.nextTime).toBe(0); expect(player.turnActive).toBe(false);
        player._getDelayMs = () => 250;
        player.beginTurn(); player.playChunk(pcm([.2, 0])); player.stopAll();
        vi.advanceTimersByTime(500);
        expect(player.ctx.sources).toHaveLength(1);
    });
    it('observes the real source start without changing its audio or target default', () => {
        const observe = vi.fn(); player.onSchedule = observe;
        player.beginTurn(); player.playChunk(pcm([0, .25, -.25]));
        const node = player.ctx.sources[0], detail = observe.mock.calls[0][0];
        expect(node.connect).toHaveBeenCalledWith(player.ctx.destination);
        expect(node.start).toHaveBeenCalledWith(detail.scheduledAudioTime);
        expect(detail).toMatchObject({samples: 3, sampleRate: 24000, outputLatencyMs: 64,
            outputContextTime: .9, outputPerformanceTime: 100});
        node.onended(); expect(node.disconnect).toHaveBeenCalledOnce();
        expect(player._sources).toHaveLength(0);
    });
    it('counts scheduling starvation, without treating intentionally silent PCM as starvation', () => {
        const gap = vi.fn(); player.onGap = gap;
        player.beginTurn(); player.playChunk(pcm(new Array(240).fill(0)));
        expect(player.gapCount).toBe(0);
        player.ctx.currentTime = 1.05; player.playChunk(pcm([0, 0]));
        vi.advanceTimersByTime(0);
        expect(player.gapCount).toBe(1);
        expect(gap.mock.calls[0][0].gap_ms).toBeCloseTo(40);
    });
    it('reports a terminal empty-queue wait even when no later packet arrives', () => {
        const metrics = vi.fn(); player.onMetrics = metrics;
        player.beginTurn(); player.playChunk(pcm(new Array(240).fill(0)));
        player.ctx.currentTime = 1.2; vi.advanceTimersByTime(200);
        player.endTurn();
        const final = metrics.mock.calls.at(-1)[0];
        expect(final.underflowMs).toBeCloseTo(190);
        expect(final.longestUnderflowMs).toBeCloseTo(190);
        expect(final.queuedAudioMs).toBe(0);
    });

});
