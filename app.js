/**
 * Malayalam Handwriting Generator - Web Demo
 * Uses ONNX Runtime for inference with Conditional Sketch-RNN
 * Simplified version: Random generation only
 */

// Model configuration
const MODEL_CONFIG = {
    vocabSize: 156,
    textDim: 512,
    latentDim: 128,
    hiddenDim: 1024,
    numMixtures: 20,
    numLayers: 1
};

// Malayalam Unicode range: U+0D00 to U+0D7F
const MALAYALAM_START = 0x0D00;
const MALAYALAM_END = 0x0D7F;

// Special tokens
const PAD_TOKEN = 0;
const UNK_TOKEN = 1;

class MalayalamHandwritingGenerator {
    constructor() {
        this.sessions = {
            textEncoder: null,
            decoderInit: null,
            decoderStep: null
        };
        this.isLoaded = false;
        this.isGenerating = false;

        // Canvas state
        this.canvas = null;
        this.ctx = null;

        // Animation state
        this.animationId = null;
        this.currentX = 0;
        this.currentY = 0;

        this.init();
    }

    async init() {
        this.setupCanvas();
        this.setupEventListeners();
        await this.loadModels();
    }

    setupCanvas() {
        this.canvas = document.getElementById('handwriting-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.clearCanvas();
    }

    setupEventListeners() {
        document.getElementById('generate-btn').addEventListener('click', () => this.generate());
        document.getElementById('clear-btn').addEventListener('click', () => this.clearCanvas());

        document.getElementById('temperature').addEventListener('input', (e) => {
            document.getElementById('temp-value').textContent = e.target.value;
        });

        document.getElementById('speed').addEventListener('input', (e) => {
            document.getElementById('speed-value').textContent = e.target.value;
        });

        document.getElementById('stroke-width').addEventListener('input', (e) => {
            document.getElementById('width-value').textContent = e.target.value;
        });
    }

    async loadModels() {
        const loading = document.getElementById('loading');

        try {
            loading.textContent = 'Loading text encoder...';
            this.sessions.textEncoder = await ort.InferenceSession.create('./onnx_models/text_encoder.onnx');

            loading.textContent = 'Loading decoder init...';
            this.sessions.decoderInit = await ort.InferenceSession.create('./onnx_models/decoder_init.onnx');

            loading.textContent = 'Loading decoder step...';
            this.sessions.decoderStep = await ort.InferenceSession.create('./onnx_models/decoder_step.onnx');

            this.isLoaded = true;
            this.updateStatus('Ready', true);
            loading.classList.add('hidden');

            console.log('All models loaded successfully');

        } catch (error) {
            console.error('Failed to load models:', error);
            this.updateStatus('Failed to load models', false);
            loading.textContent = 'Error loading models.';
        }
    }

    updateStatus(text, ready) {
        document.getElementById('status-text').textContent = text;
        document.getElementById('status-dot').classList.toggle('ready', ready);
    }

    generateRandomTokens() {
        const numTokens = Math.floor(Math.random() * 5) + 2;
        const tokens = [];

        for (let i = 0; i < numTokens; i++) {
            const charType = Math.random();
            let codePoint;

            if (charType < 0.6) {
                codePoint = 0x0D15 + Math.floor(Math.random() * (0x0D39 - 0x0D15 + 1));
            } else if (charType < 0.85) {
                codePoint = 0x0D3E + Math.floor(Math.random() * (0x0D4C - 0x0D3E + 1));
            } else {
                codePoint = 0x0D05 + Math.floor(Math.random() * (0x0D14 - 0x0D05 + 1));
            }

            tokens.push(codePoint - MALAYALAM_START + 2);
        }

        return tokens;
    }

    createTensor(data, dims, type = 'float32') {
        if (type === 'int64') {
            return new ort.Tensor('int64', BigInt64Array.from(data.map(BigInt)), dims);
        }
        return new ort.Tensor(type, Float32Array.from(data), dims);
    }

    sampleCategorical(logits, temperature = 1.0) {
        const scaledLogits = logits.map(x => x / temperature);
        const maxLogit = Math.max(...scaledLogits);
        const expLogits = scaledLogits.map(x => Math.exp(x - maxLogit));
        const sumExp = expLogits.reduce((a, b) => a + b, 0);
        const probs = expLogits.map(x => x / sumExp);

        const r = Math.random();
        let cumSum = 0;
        for (let i = 0; i < probs.length; i++) {
            cumSum += probs[i];
            if (r < cumSum) return i;
        }
        return probs.length - 1;
    }

    sampleBivariateGaussian(muX, muY, sigmaX, sigmaY, rho) {
        const z1 = this.randn();
        const z2 = this.randn();
        const x = muX + sigmaX * z1;
        const y = muY + sigmaY * (rho * z1 + Math.sqrt(1 - rho * rho) * z2);
        return [x, y];
    }

    randn() {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    sampleFromMDN(outputs, temperature = 0.4) {
        const piLogits = Array.from(outputs.pi_logits.data);
        const muX = Array.from(outputs.mu_x.data);
        const muY = Array.from(outputs.mu_y.data);
        const sigmaX = Array.from(outputs.sigma_x.data).map(s => s * temperature);
        const sigmaY = Array.from(outputs.sigma_y.data).map(s => s * temperature);
        const rho = Array.from(outputs.rho.data);
        const penLogits = Array.from(outputs.pen_logits.data);

        const k = this.sampleCategorical(piLogits, temperature);
        const [dx, dy] = this.sampleBivariateGaussian(muX[k], muY[k], sigmaX[k], sigmaY[k], rho[k]);
        const penIdx = this.sampleCategorical(penLogits, temperature);

        return {
            dx, dy,
            p1: penIdx === 0 ? 1 : 0,
            p2: penIdx === 1 ? 1 : 0,
            p3: penIdx === 2 ? 1 : 0
        };
    }

    async generate() {
        if (!this.isLoaded || this.isGenerating) return;

        this.isGenerating = true;
        this.updateStatus('Generating...', true);
        document.getElementById('generate-btn').disabled = true;

        try {
            await this.generateRandom();
        } catch (error) {
            console.error('Generation error:', error);
            this.updateStatus('Error during generation', false);
        }

        this.isGenerating = false;
        document.getElementById('generate-btn').disabled = false;
        this.updateStatus('Ready', true);
    }

    async generateRandom() {
        const temperature = parseFloat(document.getElementById('temperature').value);
        this.clearCanvas();

        const tokens = this.generateRandomTokens();
        const maxLen = 20;
        const paddedTokens = [...tokens, ...Array(maxLen - tokens.length).fill(PAD_TOKEN)].slice(0, maxLen);

        const textTensor = this.createTensor(paddedTokens, [1, maxLen], 'int64');
        const textEncResult = await this.sessions.textEncoder.run({ text_ids: textTensor });
        const textEncoding = textEncResult.text_encoding;

        const zData = Array.from({ length: MODEL_CONFIG.latentDim }, () => this.randn());
        const z = this.createTensor(zData, [1, MODEL_CONFIG.latentDim]);

        const initResult = await this.sessions.decoderInit.run({ z: z, text_encoding: textEncoding });
        let h = initResult.h0;
        let c = initResult.c0;

        let stroke = this.createTensor([0, 0, 1, 0, 0], [1, 1, 5]);
        const strokes = [];
        const maxStrokes = 300;

        for (let i = 0; i < maxStrokes; i++) {
            const stepResult = await this.sessions.decoderStep.run({
                stroke: stroke, z: z, text_encoding: textEncoding, h: h, c: c
            });

            const nextStroke = this.sampleFromMDN({
                pi_logits: stepResult.pi_logits,
                mu_x: stepResult.mu_x,
                mu_y: stepResult.mu_y,
                sigma_x: stepResult.sigma_x,
                sigma_y: stepResult.sigma_y,
                rho: stepResult.rho,
                pen_logits: stepResult.pen_logits
            }, temperature);

            strokes.push(nextStroke);
            if (nextStroke.p3 === 1) break;

            h = stepResult.h_new;
            c = stepResult.c_new;
            stroke = this.createTensor([nextStroke.dx, nextStroke.dy, nextStroke.p1, nextStroke.p2, nextStroke.p3], [1, 1, 5]);
        }

        document.getElementById('stroke-count').textContent = `Strokes: ${strokes.length}`;
        await this.animateStrokes(strokes);
    }

    async animateStrokes(strokes) {
        const speedSlider = parseInt(document.getElementById('speed').value);
        const speed = 51 - speedSlider;
        const strokeWidth = parseFloat(document.getElementById('stroke-width').value);

        this.ctx.strokeStyle = '#222';
        this.ctx.lineWidth = strokeWidth;

        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        let x = 0, y = 0;

        for (const s of strokes) {
            x += s.dx; y += s.dy;
            minX = Math.min(minX, x); maxX = Math.max(maxX, x);
            minY = Math.min(minY, y); maxY = Math.max(maxY, y);
        }

        const padding = 30;
        const availWidth = this.canvas.width - 2 * padding;
        const availHeight = this.canvas.height - 2 * padding;
        const strokeBoundsWidth = maxX - minX || 1;
        const strokeBoundsHeight = maxY - minY || 1;
        const scale = Math.min(availWidth / strokeBoundsWidth, availHeight / strokeBoundsHeight);

        this.currentX = padding + (availWidth - strokeBoundsWidth * scale) / 2 - minX * scale;
        this.currentY = padding + (availHeight - strokeBoundsHeight * scale) / 2 - minY * scale;

        let strokeIndex = 0;
        let penDown = true;
        const batchSize = speed >= 10 ? speed : Math.max(1, Math.floor(speed / 2));
        const frameDelay = speed < 10 ? (11 - speed) * 20 : 0;

        const animate = () => {
            for (let i = 0; i < batchSize && strokeIndex < strokes.length; i++, strokeIndex++) {
                const s = strokes[strokeIndex];
                const newX = this.currentX + s.dx * scale;
                const newY = this.currentY + s.dy * scale;

                if (penDown) {
                    this.ctx.beginPath();
                    this.ctx.moveTo(this.currentX, this.currentY);
                    this.ctx.lineTo(newX, newY);
                    this.ctx.stroke();
                }

                this.currentX = newX;
                this.currentY = newY;
                if (s.p2 === 1) penDown = false;
                else if (s.p1 === 1) penDown = true;
            }

            if (strokeIndex < strokes.length) {
                if (frameDelay > 0) {
                    setTimeout(() => { this.animationId = requestAnimationFrame(animate); }, frameDelay);
                } else {
                    this.animationId = requestAnimationFrame(animate);
                }
            }
        };

        animate();
    }

    clearCanvas() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        this.ctx.fillStyle = '#fff';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.currentX = this.canvas.width / 2;
        this.currentY = this.canvas.height / 2;
        document.getElementById('stroke-count').textContent = 'Strokes: 0';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.generator = new MalayalamHandwritingGenerator();
});
