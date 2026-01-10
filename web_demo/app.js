/**
 * Malayalam Handwriting Generator - Web Demo
 * Uses ONNX Runtime for inference with Conditional Sketch-RNN
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
        this.strokes = [];
        this.currentStroke = [];
        this.isDrawing = false;

        // Animation state
        this.animationId = null;
        this.strokeQueue = [];
        this.currentX = 0;
        this.currentY = 0;

        // Mode
        this.mode = 'text';

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
        // Mode selector
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.setMode(e.target.dataset.mode));
        });

        // Generate button
        document.getElementById('generate-btn').addEventListener('click', () => this.generate());

        // Clear button
        document.getElementById('clear-btn').addEventListener('click', () => this.clearCanvas());

        // Text input
        document.getElementById('text-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.generate();
        });

        // Sample words
        document.querySelectorAll('.word-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                document.getElementById('text-input').value = chip.textContent;
            });
        });

        // Sliders
        document.getElementById('temperature').addEventListener('input', (e) => {
            document.getElementById('temp-value').textContent = e.target.value;
        });

        document.getElementById('speed').addEventListener('input', (e) => {
            document.getElementById('speed-value').textContent = e.target.value;
        });

        document.getElementById('stroke-width').addEventListener('input', (e) => {
            document.getElementById('width-value').textContent = e.target.value;
        });

        // Canvas drawing events (for autocomplete mode)
        this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        this.canvas.addEventListener('mousemove', (e) => this.draw(e));
        this.canvas.addEventListener('mouseup', () => this.stopDrawing());
        this.canvas.addEventListener('mouseleave', () => this.stopDrawing());

        // Touch events
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.startDrawing(e.touches[0]);
        });
        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            this.draw(e.touches[0]);
        });
        this.canvas.addEventListener('touchend', () => this.stopDrawing());
    }

    setMode(mode) {
        this.mode = mode;

        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });

        // Toggle visibility
        document.querySelectorAll('.text-mode').forEach(el => {
            el.classList.toggle('hidden', mode !== 'text');
        });
        document.querySelectorAll('.autocomplete-mode').forEach(el => {
            el.classList.toggle('hidden', mode !== 'autocomplete');
        });

        // Update button text
        const btn = document.getElementById('generate-btn');
        btn.textContent = mode === 'text' ? 'Generate Handwriting' : 'Complete Drawing';

        this.clearCanvas();
    }

    async loadModels() {
        const loading = document.getElementById('loading');
        const loadingText = loading.querySelector('.loading-text');
        loading.classList.remove('hidden');

        try {
            loadingText.textContent = 'Loading text encoder...';
            this.sessions.textEncoder = await ort.InferenceSession.create('../onnx_models/text_encoder.onnx');

            loadingText.textContent = 'Loading decoder init...';
            this.sessions.decoderInit = await ort.InferenceSession.create('../onnx_models/decoder_init.onnx');

            loadingText.textContent = 'Loading decoder step...';
            this.sessions.decoderStep = await ort.InferenceSession.create('../onnx_models/decoder_step.onnx');

            this.isLoaded = true;
            this.updateStatus('Ready', true);
            loading.classList.add('hidden');

            console.log('All models loaded successfully');

        } catch (error) {
            console.error('Failed to load models:', error);
            this.updateStatus('Failed to load models', false);
            loadingText.textContent = 'Error loading models. Check console.';
        }
    }

    updateStatus(text, ready) {
        document.getElementById('status-text').textContent = text;
        document.getElementById('status-dot').classList.toggle('ready', ready);
    }

    // Text encoding - convert Malayalam text to token IDs
    encodeText(text) {
        const tokens = [];

        for (const char of text) {
            const codePoint = char.codePointAt(0);

            if (codePoint >= MALAYALAM_START && codePoint <= MALAYALAM_END) {
                // Map Malayalam characters: index = codePoint - MALAYALAM_START + 2
                // +2 because 0=PAD, 1=UNK
                tokens.push(codePoint - MALAYALAM_START + 2);
            } else if (codePoint === 0x200C || codePoint === 0x200D) {
                // Zero-width non-joiner and joiner
                // Map to end of Malayalam range + offset
                tokens.push(MALAYALAM_END - MALAYALAM_START + 2 + (codePoint - 0x200C + 1));
            } else {
                tokens.push(UNK_TOKEN);
            }
        }

        return tokens;
    }

    // Create tensor from array
    createTensor(data, dims, type = 'float32') {
        if (type === 'int64') {
            return new ort.Tensor('int64', BigInt64Array.from(data.map(BigInt)), dims);
        }
        return new ort.Tensor(type, Float32Array.from(data), dims);
    }

    // Sample from softmax distribution
    sampleCategorical(logits, temperature = 1.0) {
        // Apply temperature
        const scaledLogits = logits.map(x => x / temperature);

        // Softmax
        const maxLogit = Math.max(...scaledLogits);
        const expLogits = scaledLogits.map(x => Math.exp(x - maxLogit));
        const sumExp = expLogits.reduce((a, b) => a + b, 0);
        const probs = expLogits.map(x => x / sumExp);

        // Sample
        const r = Math.random();
        let cumSum = 0;
        for (let i = 0; i < probs.length; i++) {
            cumSum += probs[i];
            if (r < cumSum) return i;
        }
        return probs.length - 1;
    }

    // Sample from bivariate Gaussian
    sampleBivariateGaussian(muX, muY, sigmaX, sigmaY, rho) {
        const z1 = this.randn();
        const z2 = this.randn();

        const x = muX + sigmaX * z1;
        const y = muY + sigmaY * (rho * z1 + Math.sqrt(1 - rho * rho) * z2);

        return [x, y];
    }

    // Standard normal random
    randn() {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    // Sample from MDN output
    sampleFromMDN(outputs, temperature = 0.4) {
        const piLogits = Array.from(outputs.pi_logits.data);
        const muX = Array.from(outputs.mu_x.data);
        const muY = Array.from(outputs.mu_y.data);
        const sigmaX = Array.from(outputs.sigma_x.data).map(s => s * temperature);
        const sigmaY = Array.from(outputs.sigma_y.data).map(s => s * temperature);
        const rho = Array.from(outputs.rho.data);
        const penLogits = Array.from(outputs.pen_logits.data);

        // Sample mixture component
        const k = this.sampleCategorical(piLogits, temperature);

        // Sample dx, dy from selected Gaussian
        const [dx, dy] = this.sampleBivariateGaussian(
            muX[k], muY[k], sigmaX[k], sigmaY[k], rho[k]
        );

        // Sample pen state
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
            if (this.mode === 'text') {
                await this.generateFromText();
            } else {
                await this.autocomplete();
            }
        } catch (error) {
            console.error('Generation error:', error);
            this.updateStatus('Error during generation', false);
        }

        this.isGenerating = false;
        document.getElementById('generate-btn').disabled = false;
        this.updateStatus('Ready', true);
    }

    async generateFromText() {
        const text = document.getElementById('text-input').value.trim();
        if (!text) return;

        const temperature = parseFloat(document.getElementById('temperature').value);

        this.clearCanvas();

        // Encode text
        const tokens = this.encodeText(text);
        console.log('Text:', text, 'Tokens:', tokens);

        // Pad tokens
        const maxLen = 20;
        const paddedTokens = [...tokens, ...Array(maxLen - tokens.length).fill(PAD_TOKEN)].slice(0, maxLen);

        // Create input tensor
        const textTensor = this.createTensor(paddedTokens, [1, maxLen], 'int64');

        // Encode text
        const textEncResult = await this.sessions.textEncoder.run({ text_ids: textTensor });
        const textEncoding = textEncResult.text_encoding;

        // Sample z from standard normal
        const zData = Array.from({ length: MODEL_CONFIG.latentDim }, () => this.randn());
        const z = this.createTensor(zData, [1, MODEL_CONFIG.latentDim]);

        // Initialize decoder
        const initResult = await this.sessions.decoderInit.run({
            z: z,
            text_encoding: textEncoding
        });

        let h = initResult.h0;
        let c = initResult.c0;

        // Start token
        let stroke = this.createTensor([0, 0, 1, 0, 0], [1, 1, 5]);

        // Generate strokes
        const strokes = [];
        const maxStrokes = 300;

        for (let i = 0; i < maxStrokes; i++) {
            // Run decoder step
            const stepResult = await this.sessions.decoderStep.run({
                stroke: stroke,
                z: z,
                text_encoding: textEncoding,
                h: h,
                c: c
            });

            // Sample next stroke
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

            // Check for end token
            if (nextStroke.p3 === 1) {
                break;
            }

            // Update hidden state
            h = stepResult.h_new;
            c = stepResult.c_new;

            // Create next input stroke
            stroke = this.createTensor(
                [nextStroke.dx, nextStroke.dy, nextStroke.p1, nextStroke.p2, nextStroke.p3],
                [1, 1, 5]
            );
        }

        console.log(`Generated ${strokes.length} stroke points`);
        document.getElementById('stroke-count').textContent = `Strokes: ${strokes.length}`;

        // Animate strokes
        await this.animateStrokes(strokes);
    }

    async autocomplete() {
        // Get user strokes
        if (this.strokes.length === 0) {
            alert('Please draw something first!');
            return;
        }

        // For autocomplete, we'd need to encode the user's strokes
        // This is a simplified version - just generate from a random z
        // A full implementation would encode the user strokes first

        const temperature = parseFloat(document.getElementById('temperature').value);

        // Use a fixed "empty" text or infer from strokes
        // For now, just use blank conditioning
        const tokens = [UNK_TOKEN]; // Single unknown token
        const maxLen = 20;
        const paddedTokens = [...tokens, ...Array(maxLen - tokens.length).fill(PAD_TOKEN)];

        const textTensor = this.createTensor(paddedTokens, [1, maxLen], 'int64');
        const textEncResult = await this.sessions.textEncoder.run({ text_ids: textTensor });
        const textEncoding = textEncResult.text_encoding;

        // Sample z
        const zData = Array.from({ length: MODEL_CONFIG.latentDim }, () => this.randn() * 0.5);
        const z = this.createTensor(zData, [1, MODEL_CONFIG.latentDim]);

        // Initialize decoder
        const initResult = await this.sessions.decoderInit.run({
            z: z,
            text_encoding: textEncoding
        });

        let h = initResult.h0;
        let c = initResult.c0;

        // Get last point of user drawing as starting position
        const lastUserStroke = this.strokes[this.strokes.length - 1];
        const lastPoint = lastUserStroke[lastUserStroke.length - 1];
        this.currentX = lastPoint?.x || this.canvas.width / 2;
        this.currentY = lastPoint?.y || this.canvas.height / 2;

        // Start with pen down
        let stroke = this.createTensor([0, 0, 1, 0, 0], [1, 1, 5]);

        // Generate continuation
        const strokes = [];
        const maxStrokes = 150;

        for (let i = 0; i < maxStrokes; i++) {
            const stepResult = await this.sessions.decoderStep.run({
                stroke: stroke,
                z: z,
                text_encoding: textEncoding,
                h: h,
                c: c
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

            stroke = this.createTensor(
                [nextStroke.dx, nextStroke.dy, nextStroke.p1, nextStroke.p2, nextStroke.p3],
                [1, 1, 5]
            );
        }

        console.log(`Generated ${strokes.length} completion strokes`);

        // Animate from current position
        await this.animateStrokes(strokes, false);
    }

    async animateStrokes(strokes, resetPosition = true) {
        const speed = parseInt(document.getElementById('speed').value);
        const strokeWidth = parseFloat(document.getElementById('stroke-width').value);

        this.ctx.strokeStyle = '#1a1a2e';
        this.ctx.lineWidth = strokeWidth;

        // Calculate bounds for scaling
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        let x = 0, y = 0;

        for (const s of strokes) {
            x += s.dx;
            y += s.dy;
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
        }

        // Scale and center
        const padding = 50;
        const availWidth = this.canvas.width - 2 * padding;
        const availHeight = this.canvas.height - 2 * padding;

        const strokeWidth2 = maxX - minX || 1;
        const strokeHeight = maxY - minY || 1;

        const scale = Math.min(availWidth / strokeWidth2, availHeight / strokeHeight, 3);

        if (resetPosition) {
            this.currentX = padding + (availWidth - strokeWidth2 * scale) / 2 - minX * scale;
            this.currentY = padding + (availHeight - strokeHeight * scale) / 2 - minY * scale;
        }

        // Animate
        let strokeIndex = 0;
        let penDown = true;

        const animate = () => {
            const batchSize = speed;

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

                // Update pen state for next stroke
                if (s.p2 === 1) {
                    penDown = false; // Pen lift
                } else if (s.p1 === 1) {
                    penDown = true; // Pen down
                }
                // p3 = end of sketch
            }

            if (strokeIndex < strokes.length) {
                this.animationId = requestAnimationFrame(animate);
            }
        };

        animate();
    }

    // Canvas drawing methods for autocomplete mode
    startDrawing(e) {
        if (this.mode !== 'autocomplete') return;

        this.isDrawing = true;
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (this.canvas.width / rect.width);
        const y = (e.clientY - rect.top) * (this.canvas.height / rect.height);

        this.currentStroke = [{ x, y }];

        this.ctx.beginPath();
        this.ctx.moveTo(x, y);

        const strokeWidth = parseFloat(document.getElementById('stroke-width').value);
        this.ctx.lineWidth = strokeWidth;
        this.ctx.strokeStyle = '#1a1a2e';
    }

    draw(e) {
        if (!this.isDrawing || this.mode !== 'autocomplete') return;

        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (this.canvas.width / rect.width);
        const y = (e.clientY - rect.top) * (this.canvas.height / rect.height);

        this.currentStroke.push({ x, y });

        this.ctx.lineTo(x, y);
        this.ctx.stroke();
    }

    stopDrawing() {
        if (!this.isDrawing) return;

        this.isDrawing = false;
        if (this.currentStroke.length > 0) {
            this.strokes.push([...this.currentStroke]);
            this.currentStroke = [];
        }
    }

    clearCanvas() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }

        this.ctx.fillStyle = '#fff';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        this.strokes = [];
        this.currentStroke = [];
        this.currentX = this.canvas.width / 2;
        this.currentY = this.canvas.height / 2;

        document.getElementById('stroke-count').textContent = 'Strokes: 0';
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    window.generator = new MalayalamHandwritingGenerator();
});
